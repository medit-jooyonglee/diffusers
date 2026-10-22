"""FLUX.2 klein-4B(4 step)로 사람/환경 이미지를 생성(text2img)하고, 편집 프롬프트로 편집(img2edit)해 저장한다.

저장 구조 (같은 id끼리 짝, 하위 폴더는 자동 생성):
  <output-dir>/text2img/<id>.png    생성한 원본
  <output-dir>/text2img/<id>.txt    text2img 프롬프트
  <output-dir>/editimg/<id>.png     편집 결과
  <output-dir>/editimg/<id>.txt     편집 프롬프트
  <output-dir>/editimg/<id>.json    메타 (입력 이미지 파일명, 편집 프롬프트, seed, 크기, step)

생성 프롬프트(인물·스타일·환경)와 편집 프롬프트는 아래 목록에서 seed 기준으로 무작위 조합한다 — 같은 --seed면 같은 조합이 나온다.
editimg의 메타(json)까지 있는 id는 건너뛴다(중단 후 다시 실행하면 이어서 생성).

예:
  python my/flux2_klein_t2i_edit_dataset.py --num-images 8 --output-dir outputs/klein_t2i_edit
  python my/flux2_klein_t2i_edit_dataset.py --num-images 10000 --output-dir /data1/.../klein_edit_v1 \\
      --fp8-compute --compile regional --num-shards 2 --shard-index 0 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flux2_klein_edit_batch import DEFAULT_MODEL, DEFAULT_TEXT_ENCODER_MODEL, load_pipeline  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402


# (나이 표현, 그룹) — 그룹이 명사(girl/boy, teenage girl/boy, woman/man)와 고를 수 있는 편집을 정한다
AGES = [("5-year-old", "child"), ("10-year-old", "child"), ("teenage", "teen"), ("young adult", "adult"), ("thirty-something", "adult"),
        ("middle-aged", "adult"), ("fifty-something", "adult"), ("elderly", "senior"), ("very old", "senior")]
NOUNS = {"child": {"female": "girl", "male": "boy"}, "teen": {"female": "girl", "male": "boy"},
         "adult": {"female": "woman", "male": "man"}, "senior": {"female": "woman", "male": "man"}}
SEXES = ["female", "male"]
ALL, GROWN = {"child", "teen", "adult", "senior"}, {"adult", "senior"}
ETHNICITIES = ["East Asian", "Korean", "Japanese", "South Asian", "Southeast Asian", "Black", "White", "Hispanic", "Middle Eastern",
               "Native American", "Pacific Islander"]
HAIR_STYLES = ["short curly", "long straight", "bob-cut", "braided", "wavy shoulder-length", "messy bun", "pixie-cut", "afro",
               "ponytail", "slicked-back", "buzz-cut", "layered medium-length", "dreadlocked", "side-parted", "spiky", "long wavy"]
HAIR_COLORS = ["black", "dark brown", "light brown", "blonde", "auburn", "gray", "white", "red"]
OUTFITS = ["a denim jacket and white t-shirt", "a tailored navy suit", "a cozy knit sweater", "a summer floral dress", "a leather jacket",
           "a hiking outfit with a backpack", "a chef's uniform", "a hoodie and jeans", "a traditional hanbok", "a white lab coat",
           "a business blazer", "a sports tracksuit", "a striped polo shirt", "a wool winter coat and scarf", "a linen summer shirt",
           "overalls over a plaid shirt", "a black turtleneck", "a nurse uniform", "a firefighter uniform", "a school uniform",
           "a yellow raincoat", "a graphic t-shirt", "a cardigan over a blouse", "a cycling jersey", "a kimono", "a sari",
           "a flannel shirt", "a puffer jacket", "a silk evening gown", "a mechanic's work overalls", "a cozy pajama set",
           "a beige trench coat", "a varsity jacket", "a sleeveless running top", "a tweed jacket", "a festival outfit with beads"]
ACTIONS = ["smiling at the camera", "laughing naturally", "looking thoughtfully to the side", "walking toward the camera",
           "sitting and reading a book", "holding a cup of coffee", "waving with one hand", "standing with arms crossed",
           "leaning against a wall", "talking on a phone", "typing on a laptop", "carrying shopping bags", "riding a bicycle",
           "playing a guitar", "cooking at a stove", "watering plants", "taking a photo with a camera", "stretching after a run",
           "sitting on a bench", "eating an ice cream", "holding an umbrella", "looking out of a window", "painting on a canvas",
           "tying their shoelaces", "dancing", "hugging a dog"]
ENVIRONMENTS = ["a busy city street at golden hour", "a quiet forest trail", "a modern minimalist office", "a sunny beach",
                "a cozy cafe with warm lights", "a snowy mountain village", "a rainy neon-lit alley at night", "a colorful flower market",
                "a library with tall bookshelves", "a rooftop with a city skyline", "a desert road", "a kitchen at home",
                "a subway platform", "a farmers market", "a university campus", "a hospital corridor", "a fishing harbor",
                "a rice field in the countryside", "a traditional Korean palace courtyard", "a Japanese temple garden",
                "an industrial warehouse", "a cherry blossom park", "a lakeside dock at dawn", "a crowded music festival",
                "a classroom", "a gym with weights", "a bakery counter", "a train station", "an art gallery", "a camping site with a tent",
                "a vineyard", "a living room with a sofa", "a parking garage", "a tropical jungle", "an autumn park with fallen leaves",
                "a bookstore", "a laboratory", "a basketball court", "a night market with lanterns", "a mountain summit",
                "a greenhouse full of plants", "a hotel lobby", "an airport terminal", "a garden patio", "a recording studio",
                "a pottery workshop", "a seaside cliff", "a snowy city street"]
STYLES = ["a photorealistic portrait photograph, 50mm lens, natural skin texture",
          "a cinematic film still with shallow depth of field",
          "a candid documentary street photograph",
          "a fashion editorial photograph",
          "a vintage 35mm film photograph with soft grain",
          "a high-resolution DSLR photograph",
          "a smartphone snapshot",
          "a black-and-white photograph",
          "a polaroid photograph",
          "a soft pastel illustration",
          "a detailed digital painting",
          "a 3D animated movie style render",
          "an anime illustration",
          "a watercolor painting",
          "an oil painting",
          "a comic book illustration",
          "a flat vector illustration",
          "a realistic pencil drawing"]
SHOTS = ["a close-up portrait", "a head-and-shoulders portrait", "a half-body shot", "a three-quarter body shot", "a full-body shot",
         "a wide environmental portrait", "a medium shot"]
LIGHTING = ["soft natural daylight", "warm golden-hour sunlight", "overcast diffused light", "dramatic side lighting", "neon lights",
            "warm indoor lamp light", "bright studio lighting", "moonlight", "backlight with a soft rim", "dappled light through trees"]
ANGLES = ["at eye level", "from a slightly low angle", "from a slightly high angle", "from the side", "from a three-quarter view"]

EDITS = [  # (문장, 성별 조건, 나이 그룹 조건)
    ("Change the hair color to {hair_color}.", None, ALL),
    ("Change the hairstyle to {hairstyle}.", None, ALL),
    ("Change the outfit to {outfit}.", None, ALL),
    ("Change the color of the clothes to {color}.", None, ALL),
    ("Replace the background with {environment}. Keep the person exactly the same.", None, ALL),
    ("Make the person {expression}.", None, ALL),
    ("Add {accessory} to the person.", None, ALL),
    ("Make the person hold {object}.", None, ALL),
    ("Change the pose so the person is {pose}.", None, ALL),
    ("Change the time of day to {time}.", None, ALL),
    ("Change the weather to {weather}.", None, ALL),
    ("Change the season to {season}.", None, ALL),
    ("Turn the image into {style}. Keep the composition and the person the same.", None, ALL),
    ("Make the lighting {lighting}.", None, ALL),
    ("Apply {grade} color grading to the whole image.", None, ALL),
    ("Make the person look about {years} years older.", None, {"child", "teen", "adult"}),
    ("Make the person look about {years} years younger.", None, GROWN),
    ("Add {makeup} to the person's face.", "female", {"teen", "adult", "senior"}),
    ("Give the man {beard}.", "male", GROWN),
    ("Add {pet} next to the person.", None, ALL),
    ("Blur the background like a portrait photo with a wide-aperture lens.", None, ALL),
    ("Make the scene look like it is from {era}.", None, ALL),
    ("Add {tattoo} on the person's forearm.", None, GROWN),
    ("Change the pattern of the clothes to {pattern}.", None, ALL),
    ("Replace the floor or ground with {ground}.", None, ALL),
    ("Add {effect} to the scene.", None, ALL),
    ("Make the person wear {headwear}.", None, ALL),
    ("Change the person's eye color to {eye_color}.", None, ALL),
    ("Zoom out to show more of the surroundings while keeping the person the same.", None, ALL),
]
FILL = {
    "hairstyle": ["a short pixie cut", "long wavy hair", "a sleek high ponytail", "a curly afro", "a neat side part", "two braids",
                  "a buzz cut", "a voluminous blowout", "a top knot bun", "shoulder-length straight hair", "a mullet", "bangs with a bob"],
    "outfit": ["a black tuxedo", "a yellow raincoat", "a white wedding dress", "a red sports jersey", "a green military jacket",
               "a beige trench coat", "a Hawaiian shirt", "a medieval knight armor", "an astronaut suit", "a chef's uniform",
               "a business suit with a tie", "a denim overall", "a ski jacket", "a traditional hanbok", "a superhero costume",
               "a doctor's white coat", "a pink hoodie", "a leather biker jacket", "a tennis outfit", "a tuxedo-style vest"],
    "color": ["bright red", "navy blue", "emerald green", "mustard yellow", "pure white", "all black", "lavender", "orange", "teal",
              "burgundy", "pastel pink", "olive green"],
    "environment": ["a tropical beach at sunset", "a snowy forest", "a busy Tokyo street at night", "a plain white studio backdrop",
                    "a medieval castle courtyard", "a field of sunflowers", "the surface of the moon", "an underwater coral reef",
                    "a cozy library", "a futuristic sci-fi city", "a lavender field", "a rainy London street", "a desert with dunes",
                    "a bamboo forest", "a Paris cafe terrace", "a mountain lake", "a busy office", "a night sky full of stars",
                    "an autumn forest", "a Seoul street with signboards"],
    "expression": ["smile widely", "look surprised", "look serious", "close their eyes and relax", "laugh", "look sad", "wink",
                   "look angry", "look confident", "look sleepy", "smile softly with closed lips", "look shocked"],
    "accessory": ["round glasses", "a wide-brimmed hat", "a red scarf", "headphones", "a pearl necklace", "sunglasses", "a backpack",
                  "a wristwatch", "gold hoop earrings", "a bow tie", "a face mask", "a flower crown", "a baseball cap", "a camera strap"],
    "object": ["a bouquet of flowers", "a coffee cup", "an open book", "a smartphone", "a red umbrella", "a basketball", "a guitar",
               "a balloon", "a small potted plant", "a laptop", "a slice of pizza", "a teddy bear", "a lantern", "a paint brush"],
    "pose": ["sitting on a chair", "waving at the camera", "pointing to the side", "giving a thumbs up", "crossing their arms",
             "putting hands in pockets", "jumping in the air", "looking over their shoulder", "resting their chin on one hand"],
    "time": ["night with streetlights", "early morning with soft mist", "sunset with warm orange light", "bright noon", "blue hour dusk",
             "midnight under moonlight"],
    "weather": ["heavy rain", "light snowfall", "a foggy day", "a clear sunny day", "a thunderstorm", "a windy day with blowing leaves",
                "a heavy blizzard"],
    "season": ["winter with snow on the ground", "spring with cherry blossoms", "summer with lush green trees", "autumn with red and orange leaves"],
    "style": ["a watercolor painting", "a pencil sketch", "an oil painting", "a comic book illustration", "a black-and-white photograph",
              "an anime illustration", "a 3D clay animation style", "a pixel art image", "a Van Gogh style painting", "a stained glass artwork",
              "a ukiyo-e woodblock print", "a cyberpunk digital art", "a children's book illustration", "a vintage poster"],
    "lighting": ["dramatic with strong side light", "soft and diffused", "warm and golden", "cool blue", "neon pink and blue",
                 "low-key and moody", "bright and airy", "candlelit"],
    "grade": ["a warm vintage", "a cool teal-and-orange cinematic", "a faded film", "a high-contrast", "a muted pastel", "a sepia"],
    "years": ["10", "20", "30", "40"],
    "makeup": ["red lipstick", "natural everyday makeup", "smoky eye makeup", "glitter eye shadow", "rosy blush", "bold winged eyeliner"],
    "beard": ["a full beard", "a short stubble", "a mustache", "a goatee"],
    "pet": ["a golden retriever", "a small white cat", "a corgi", "a parrot on the shoulder", "a rabbit", "a black labrador"],
    "era": ["the 1920s", "the 1950s", "the 1970s", "the 1980s", "the 1990s", "the Victorian era", "a futuristic year 2200"],
    "tattoo": ["a small rose tattoo", "a geometric line tattoo", "a dragon tattoo", "a minimalist wave tattoo"],
    "pattern": ["red and white stripes", "polka dots", "a floral print", "a checkered pattern", "camouflage", "leopard print"],
    "ground": ["a sandy beach", "fresh green grass", "a wooden deck", "a wet reflective street", "a marble floor", "a carpet of autumn leaves"],
    "effect": ["falling cherry blossom petals", "floating soap bubbles", "colorful confetti", "drifting fog", "fireflies", "falling snowflakes",
               "a rainbow in the sky", "sparkling fireworks in the sky"],
    "headwear": ["a knitted beanie", "a cowboy hat", "a crown", "a chef's hat", "a bucket hat", "a motorcycle helmet", "a beret"],
    "eye_color": ["bright blue", "green", "amber", "gray"],
}
COMPOUND_PROBABILITY = 0.2  # 편집 두 가지를 한 문장으로
SIZES = [(1024, 1024), (832, 1216), (1216, 832), (896, 1152), (1152, 896)]


def text2img_prompt(rng: random.Random) -> tuple[str, dict]:
    age, group = rng.choice(AGES)
    attrs = {"sex": rng.choice(SEXES), "group": group, "hair_color": rng.choice(HAIR_COLORS)}
    words = f"{age} {rng.choice(ETHNICITIES)} {NOUNS[group][attrs['sex']]}"
    person = f"{'an' if words[0] in 'aeiou' else 'a'} {words}"
    style = rng.choice(STYLES)
    prompt = (f"{style[0].upper() + style[1:]}: {rng.choice(SHOTS)} of {person} with {rng.choice(HAIR_STYLES)} "
              f"{attrs['hair_color']} hair, wearing {rng.choice(OUTFITS)}, {rng.choice(ACTIONS)}, in {rng.choice(ENVIRONMENTS)}, "
              f"{rng.choice(LIGHTING)}, photographed {rng.choice(ANGLES)}.")
    return prompt, attrs


def _one_edit(rng: random.Random, attrs: dict, exclude: set) -> str:
    candidates = [t for t, sex, groups in EDITS
                  if (sex is None or sex == attrs["sex"]) and attrs["group"] in groups and t not in exclude]
    template = rng.choice(candidates)
    exclude.add(template)
    values = {key: rng.choice(options) for key, options in FILL.items()}
    values["hair_color"] = rng.choice([c for c in HAIR_COLORS + ["platinum blonde", "pastel pink", "deep blue", "bright red", "silver"]
                                       if c != attrs["hair_color"]])  # 원래 머리색과 같은 색으로 바꾸라는 지시는 피한다
    return template.format(**values)


def edit_prompt(rng: random.Random, attrs: dict) -> str:
    used: set = set()
    first = _one_edit(rng, attrs, used)
    if rng.random() < COMPOUND_PROBABILITY:
        return f"{first} {_one_edit(rng, attrs, used)}"
    return first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-images", type=int, required=True, help="생성할 (원본, 편집) 쌍의 수 (전체 shard 합계)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="프롬프트 조합과 생성 seed의 기준값")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-encoder-model", default=DEFAULT_TEXT_ENCODER_MODEL)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp8-weights", action=argparse.BooleanOptionalAction, default=True,
                        help="transformer 가중치 fp8 저장 (24GB GPU에서는 끄면 OOM이 날 수 있다)")
    parser.add_argument("--fp8-compute", action="store_true", help="torchao fp8 연산 (--compile과 함께 써야 빨라진다)")
    parser.add_argument("--compile", nargs="?", const="full", choices=("full", "regional"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


def main() -> None:
    args = parse_args()
    t2i_dir, edit_dir = args.output_dir / "text2img", args.output_dir / "editimg"
    t2i_dir.mkdir(parents=True, exist_ok=True)
    edit_dir.mkdir(parents=True, exist_ok=True)
    indices = [i for i in range(args.num_images) if i % args.num_shards == args.shard_index]
    todo = [i for i in indices if not (edit_dir / f"{i:06d}.json").exists()]  # json을 마지막에 쓰므로 완료 표시로 쓴다
    print(f"{len(indices)} pairs in shard {args.shard_index}/{args.num_shards}, {len(todo)} to run", flush=True)
    if not todo:
        return
    # HF 캐시 경로(.../models--org--name/snapshots/<hash>)면 org/name으로 기록
    model_name = next((part[len("models--"):].replace("--", "/") for part in Path(args.model).parts if part.startswith("models--")), args.model)
    started = time.time()
    pipeline = load_pipeline(args)  # guidance 1.0: distilled 모델은 CFG를 쓰지 않는다
    print(f"pipeline loaded in {time.time() - started:.1f}s", flush=True)

    with torch.inference_mode():
        for count, index in enumerate(todo, 1):
            sample_id = f"{index:06d}"
            rng = random.Random(f"{args.seed}-{index}")  # 샘플마다 독립적으로 재현 가능
            width, height = rng.choice(SIZES)
            source_prompt, attrs = text2img_prompt(rng)
            instruction = edit_prompt(rng, attrs)
            generator = lambda offset: torch.Generator(device=args.device).manual_seed(args.seed * 1_000_003 + index * 2 + offset)
            try:
                t0 = time.time()
                source_path = t2i_dir / f"{sample_id}.png"
                if source_path.exists():
                    source = Image.open(source_path).convert("RGB")
                else:
                    source = pipeline(prompt=source_prompt, height=height, width=width, num_inference_steps=args.steps,
                                      guidance_scale=1.0, generator=generator(0)).images[0]
                    source.save(source_path)
                if not (t2i_dir / f"{sample_id}.txt").exists():
                    (t2i_dir / f"{sample_id}.txt").write_text(source_prompt + "\n", encoding="utf-8")
                edited = pipeline(image=[source], prompt=instruction, height=source.height, width=source.width,
                                  num_inference_steps=args.steps, guidance_scale=1.0, generator=generator(1)).images[0]
                edited.save(edit_dir / f"{sample_id}.png")
                (edit_dir / f"{sample_id}.txt").write_text(instruction + "\n", encoding="utf-8")
                meta = {"id": sample_id, "input_image": f"text2img/{sample_id}.png", "edit_prompt": instruction,
                        "width": source.width, "height": source.height, "steps": args.steps,
                        "seed": args.seed * 1_000_003 + index * 2 + 1, "model": model_name}
                (edit_dir / f"{sample_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                print(f"[{count}/{len(todo)}] {sample_id} {width}x{height} {time.time() - t0:.1f}s | {instruction}", flush=True)
            except Exception as error:  # 한 건이 실패해도 나머지는 계속 생성 (다시 실행하면 재시도)
                print(f"[{count}/{len(todo)}] {sample_id} error {type(error).__name__}: {error}", flush=True)
    print(f"done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()


# TORCHINDUCTOR_CACHE_DIR=$PWD/outputs/inductor_cache TRITON_CACHE_DIR=$PWD/outputs/inductor_cache/triton \
# nohup python my/flux2_klein_t2i_edit_dataset.py --num-images 17000 \
#   --output-dir /data1/jooyonglee/public/flux2_klein4b_t2i_edit_17k \
#   --seed 0 --fp8-compute --compile regional > logs/klein_t2i_edit_17k.log 2>&1 &
