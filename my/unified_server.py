"""Launches design_tune_server.py (FLUX.2) and the iSmileNet img2vid backend_server together.

Each server's own code and routes are kept exactly as-is (two processes, two ports, same as running
them separately). The only thing this adds is turning on the cross-process GPU lock
(gpu_sequential_lock.py, --serialize-gpu / --serialize_gpu) in both, so a request to one waits for an
in-flight request on the other instead of running concurrently and risking OOM (see
docs/deploy_arch.md and the VRAM check in outputs/unified_oom_test/). By default this is sequential
processing, not a merged endpoint: call design_tune_server's API, then unified_client.py or your own
code feeds the result into the img2vid server's API.

Example:
    python my/unified_server.py --wan-repo /workspace/repo/iSmileNet --flux-cpu-offload

    # disable the cross-process lock (each server only serializes its own requests, as before):
    python my/unified_server.py --wan-repo /workspace/repo/iSmileNet --no-serialize-gpu
"""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_flux_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, str(ROOT / "my" / "design_tune_server.py"),
        "--host", args.host, "--port", str(args.flux_port), "--device", args.device,
    ]
    if args.flux_cpu_offload:
        command.append("--cpu-offload")
    if args.serialize_gpu:
        command.append("--serialize-gpu")
    if args.flux_extra_args:
        command.extend(shlex.split(args.flux_extra_args))
    return command


def build_wan_command(args: argparse.Namespace) -> list[str]:
    backend_server = Path(args.wan_repo) / "smiledesign" / "img2vid" / "backend_server..py"
    if not backend_server.is_file():
        raise FileNotFoundError(f"img2vid backend_server not found: {backend_server}")
    command = [
        args.wan_python, str(backend_server),
        "--host", args.host, "--port", str(args.wan_port), "--device_id", str(args.wan_device_id),
    ]
    if args.serialize_gpu:
        command.append("--serialize_gpu")
    if args.wan_extra_args:
        command.extend(shlex.split(args.wan_extra_args))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--flux-port", type=int, default=5000)
    parser.add_argument("--wan-port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0", help="design_tune_server --device")
    parser.add_argument("--wan-device-id", type=int, default=0, help="img2vid backend_server --device_id")
    parser.add_argument("--flux-cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--serialize-gpu", action=argparse.BooleanOptionalAction, default=True,
        help="Cross-process GPU lock between the two servers (gpu_sequential_lock.py). On by default "
        "here since that is the point of running them together; pass --no-serialize-gpu to go back to "
        "each server only serializing its own requests.",
    )
    parser.add_argument(
        "--wan-repo", default=str(ROOT.parent / "iSmileNet"),
        help="Path to the iSmileNet repo containing smiledesign/img2vid/backend_server..py",
    )
    parser.add_argument("--wan-python", default=sys.executable, help="Interpreter for the img2vid backend_server")
    parser.add_argument(
        "--flux-extra-args", default="",
        help="Extra args appended as-is to design_tune_server.py, e.g. '--model /path --fp8-weights'",
    )
    parser.add_argument(
        "--wan-extra-args", default="",
        help="Extra args appended as-is to backend_server..py, e.g. '--models wan --transformer_dir /path'",
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "outputs" / "unified_server_logs")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    flux_log = open(args.log_dir / "design_tune_server.log", "w")
    wan_log = open(args.log_dir / "backend_server.log", "w")

    flux_command = build_flux_command(args)
    wan_command = build_wan_command(args)
    print(f"[unified_server] serialize_gpu={args.serialize_gpu}")
    print(f"[unified_server] flux : {' '.join(flux_command)}")
    print(f"[unified_server] wan  : {' '.join(wan_command)}")

    flux_proc = subprocess.Popen(flux_command, cwd=ROOT, stdout=flux_log, stderr=subprocess.STDOUT)
    wan_proc = subprocess.Popen(
        wan_command, cwd=Path(args.wan_repo) / "smiledesign" / "img2vid", stdout=wan_log, stderr=subprocess.STDOUT
    )
    procs = {"design_tune_server": flux_proc, "img2vid backend_server": wan_proc}
    print(f"[unified_server] flux pid={flux_proc.pid} log={flux_log.name}")
    print(f"[unified_server] wan  pid={wan_proc.pid} log={wan_log.name}")

    stopping = False

    def shutdown(signum=None, frame=None):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print("[unified_server] stopping both servers ...")
        for name, proc in procs.items():
            if proc.poll() is None:
                proc.terminate()
        for name, proc in procs.items():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print(f"[unified_server] {name} did not exit in time, killing")
                proc.kill()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while not stopping:
            for name, proc in procs.items():
                code = proc.poll()
                if code is not None:
                    print(f"[unified_server] {name} exited with code {code}; stopping the other server")
                    shutdown()
                    sys.exit(code or 1)
            time.sleep(2)
    finally:
        flux_log.close()
        wan_log.close()


if __name__ == "__main__":
    main()
