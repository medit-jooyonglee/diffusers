"""design_tune_server.py 테스트용 커맨드라인 클라이언트.

예시:
    python my/design_tune_client.py health
    python my/design_tune_client.py styles
    python my/design_tune_client.py build smile_v1 --layout patient_only --wait
    python my/design_tune_client.py mask patient.png --regions lips,teeth --overlay -o mask.png
    python my/design_tune_client.py infer smile_v1 patient.png --preset strong_whitening --mask lips,teeth -o out.png

    # 빌드 요청 -> 완료 대기 -> 폴더 이미지를 순서대로 추론 (프리셋 여러 개면 이미지마다 전부)
    python my/design_tune_client.py run smile_v1 my/samples/faces/samples outputs/design_tune_run \\
        --build --layout patient_only --preset natural,strong_whitening --mask lips,teeth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


DEFAULT_SERVER_URL = os.environ.get("DIFFUSION_TUNE_SERVER_URL", "http://127.0.0.1:5000")


def fail(response: requests.Response) -> None:
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    sys.exit(f"HTTP {response.status_code}: {detail}")


def print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def mask_options(args: argparse.Namespace) -> dict | bool | None:
    if args.no_mask:
        return False
    if args.mask is None:
        return None  # 서버 기본값 사용
    options = {"regions": args.mask}
    if args.dilate is not None:
        options["dilate"] = args.dilate
    if args.feather is not None:
        options["feather"] = args.feather
    return options


def post_images(url: str, images: list[Path], parameters: dict, timeout: float) -> requests.Response:
    files = [("images", (path.name, path.read_bytes(), "application/octet-stream")) for path in images]
    return requests.post(url, data={"parameters": json.dumps(parameters, ensure_ascii=False)}, files=files, timeout=timeout)


def cmd_health(args: argparse.Namespace) -> None:
    response = requests.get(f"{args.server}/health", timeout=10)
    if not response.ok:
        fail(response)
    print_json(response.json())


def cmd_styles(args: argparse.Namespace) -> None:
    response = requests.get(f"{args.server}/api/v1/styles", timeout=30)
    if not response.ok:
        fail(response)
    for style in response.json()["styles"]:
        print(f"{style['name']:20s} ready={style.get('library_ready')}  presets={style.get('presets')}")


def build(server: str, style: str, layouts: list[str] | None, overwrite: bool, wait: bool) -> None:
    body = {"overwrite": overwrite}
    if layouts:
        body["layouts"] = layouts
    response = requests.post(f"{server}/api/v1/styles/{style}/build", json=body, timeout=30)
    if not response.ok:
        fail(response)
    job = response.json()
    if job.get("skipped"):
        print(f"{style}: already built for the current YAML content (skipped, use --overwrite to rebuild)")
        return
    print(f"build job {job['job_id']} {job['status']}")
    started = time.time()
    while wait and job["status"] in {"queued", "running"}:
        time.sleep(2)
        job = requests.get(f"{server}/api/v1/jobs/{job['job_id']}", timeout=30).json()
        print(f"  {time.time() - started:5.0f}s {job['status']}")
    if job["status"] == "failed":
        print(job.get("error"), job.get("log", "")[-2000:], sep="\n")
        sys.exit(1)


def cmd_build(args: argparse.Namespace) -> None:
    build(args.server, args.style, args.layout, args.overwrite, args.wait)


def cmd_mask(args: argparse.Namespace) -> None:
    parameters = {"overlay": args.overlay}
    options = mask_options(args)
    if isinstance(options, dict):
        parameters["mask"] = options
    response = post_images(f"{args.server}/api/v1/mask", [args.image], parameters, timeout=60)
    if not response.ok:
        fail(response)
    args.output.write_bytes(response.content)
    print(f"saved {args.output}")
    print_json(json.loads(response.headers["X-Design-Tune-Mask"]))


def cmd_infer(args: argparse.Namespace) -> None:
    parameters = {"style_name": args.style, "layout": args.layout, "seed": args.seed}
    if args.preset:
        parameters.update(mode="preset", preset=args.preset)
    else:
        parameters.update(mode=args.mode, control_values=json.loads(args.controls))
        if args.grid:
            parameters["grid"] = args.grid
        if args.attribute:
            parameters["attribute"] = args.attribute
    if args.steps is not None:
        parameters["num_steps"] = args.steps
    options = mask_options(args)
    if options is not None:
        parameters["mask"] = options

    started = time.time()
    response = post_images(f"{args.server}/api/v1/infer", args.images, parameters, timeout=600)
    if not response.ok:
        fail(response)
    args.output.write_bytes(response.content)
    print(f"saved {args.output} ({time.time() - started:.1f}s)")
    print_json(json.loads(response.headers["X-Design-Tune-Metadata"]))


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def cmd_run(args: argparse.Namespace) -> None:
    if args.build:
        build(args.server, args.style, [args.layout], args.overwrite, wait=True)
    images = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        sys.exit(f"No images in {args.input_dir}")
    references = [Path(path) for path in args.reference or []]
    presets = [name.strip() for name in args.preset.split(",") if name.strip()]
    options = mask_options(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, image in enumerate(images, 1):
        for preset in presets:
            parameters = {"style_name": args.style, "layout": args.layout, "mode": "preset", "preset": preset, "seed": args.seed}
            if args.steps is not None:
                parameters["num_steps"] = args.steps
            if options is not None:
                parameters["mask"] = options
            output = args.output_dir / f"{image.stem}__{preset}.png"
            started = time.time()
            response = post_images(f"{args.server}/api/v1/infer", [image, *references], parameters, timeout=600)
            elapsed = time.time() - started
            if response.ok:
                output.write_bytes(response.content)
                metadata = json.loads(response.headers["X-Design-Tune-Metadata"])
                status = "ok"
            else:
                # 얼굴 미검출 등 한 장이 실패해도 나머지는 계속 처리
                try:
                    metadata = {"error": response.json().get("error")}
                except ValueError:
                    metadata = {"error": response.text or response.reason}
                status = f"HTTP {response.status_code}"
            print(f"[{index}/{len(images)}] {image.name} {preset}: {status} {elapsed:.1f}s")
            results.append({"image": str(image), "preset": preset, "status": status, "seconds": round(elapsed, 2),
                            "output": str(output) if response.ok else None, "metadata": metadata})

    summary = args.output_dir / "results.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    succeeded = sum(item["status"] == "ok" for item in results)
    print(f"done {succeeded}/{len(results)} succeeded, summary: {summary}")


def add_mask_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mask", help="Comma-separated regions, e.g. lips,teeth (omit to use the server default)")
    parser.add_argument("--no-mask", action="store_true", help="Force the mask off for this request")
    parser.add_argument("--dilate", type=int)
    parser.add_argument("--feather", type=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default=DEFAULT_SERVER_URL)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("health").set_defaults(func=cmd_health)
    commands.add_parser("styles").set_defaults(func=cmd_styles)

    build = commands.add_parser("build")
    build.add_argument("style")
    build.add_argument("--layout", action="append", help="Repeatable; omit to build every layout")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--wait", action="store_true", help="Poll until the build job finishes")
    build.set_defaults(func=cmd_build)

    mask = commands.add_parser("mask")
    mask.add_argument("image", type=Path)
    mask.add_argument("--regions", dest="mask", default="inner_mouth")
    mask.add_argument("--dilate", type=int)
    mask.add_argument("--feather", type=float)
    mask.add_argument("--overlay", action="store_true")
    mask.add_argument("-o", "--output", type=Path, default=Path("mask.png"))
    mask.set_defaults(func=cmd_mask, no_mask=False)

    infer = commands.add_parser("infer")
    infer.add_argument("style")
    infer.add_argument("images", type=Path, nargs="+", help="Patient image first, then references in layout order")
    infer.add_argument("--layout", default="patient_only")
    infer.add_argument("--preset", help="Use preset mode with this preset name")
    infer.add_argument("--mode", default="grid", choices=("anchor", "grid", "delta"))
    infer.add_argument("--controls", default="{}", help='JSON, e.g. \'{"whitening": 70}\'')
    infer.add_argument("--grid")
    infer.add_argument("--attribute")
    infer.add_argument("--steps", type=int)
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("-o", "--output", type=Path, default=Path("design_tune.png"))
    add_mask_arguments(infer)
    infer.set_defaults(func=cmd_infer)

    run = commands.add_parser("run", help="Optionally build, then infer every image in a folder one by one")
    run.add_argument("style")
    run.add_argument("input_dir", type=Path)
    run.add_argument("output_dir", type=Path)
    run.add_argument("--build", action="store_true", help="Request an embedding build for --layout and wait first")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--layout", default="patient_only")
    run.add_argument("--reference", action="append", help="Extra reference image after the patient (repeatable, layout order)")
    run.add_argument("--preset", default="natural", help="Comma-separated presets; each image is run once per preset")
    run.add_argument("--steps", type=int)
    run.add_argument("--seed", type=int, default=0)
    add_mask_arguments(run)
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.server = args.server.rstrip("/")
    args.func(args)


if __name__ == "__main__":
    main()
