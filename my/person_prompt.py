import argparse
import hashlib
import json
import random
from pathlib import Path


ETHNICITIES = [
    "East Asian",
    "White",
    "Black",
    "South Asian",
    "Hispanic or Latino",
    "Middle Eastern",
]

AGE_GROUPS = {
    "young_child": range(3, 8),
    "older_child": range(8, 13),
    "young_adult": range(18, 29),
    "middle_aged": range(35, 51),
    "older_adult": range(65, 81),
}

SEXES = ["female", "male"]

HAIR_DETAILS = [
    "with short straight hair",
    "with shoulder-length wavy hair",
    "with tightly curled hair",
    "with long braided hair",
    "with a neat cropped hairstyle",
    "with their hair tied back",
    "with a textured layered hairstyle",
]

OLDER_HAIR_DETAILS = [
    "with natural silver hair",
    "with short gray hair",
    "with gray-streaked wavy hair",
]

EXPRESSIONS = [
    "with a calm neutral expression and eyes looking at the camera",
    "with a gentle smile and a relaxed gaze",
    "with a focused expression and gaze directed slightly downward",
    "laughing naturally with relaxed facial muscles",
    "with a thoughtful expression and gaze directed slightly off camera",
    "with a confident expression and natural eye contact",
]

OUTFITS = {
    "child": [
        "a soft knitted sweater, straight-leg trousers, and comfortable sneakers",
        "a colorful hooded jacket, denim jeans, and lace-up shoes",
        "a cotton T-shirt layered under lightweight overalls",
        "a weather-appropriate coat, wool scarf, trousers, and boots",
        "a casual tracksuit and running shoes",
        "a linen shirt, relaxed-fit trousers, and canvas shoes",
    ],
    "adult": [
        "a textured knitted sweater, denim jeans, and leather shoes",
        "a black leather jacket over a plain cotton shirt and dark trousers",
        "a tailored navy suit with a crisp shirt and polished shoes",
        "casual athletic clothing and running sneakers",
        "a linen button-down shirt with rolled sleeves and relaxed trousers",
        "a structured winter coat with a wool scarf and boots",
        "a denim jacket over a ribbed shirt with straight-leg trousers",
        "a practical work jacket, cotton shirt, and durable trousers",
    ],
}

OUTDOOR_LIGHTINGS = [
    "soft golden-hour side light with natural skin highlights",
    "overcast outdoor light with even facial exposure",
    "direct midday sunlight with crisp, physically consistent shadows",
    "nighttime neon reflections with controlled contrast and visible skin detail",
    "soft rim light separating the subject from the background",
]

INDOOR_LIGHTINGS = [
    "diffused daylight from a large window with gentle shadows",
    "warm practical lighting mixed with soft window light",
    "soft directional window light with realistic falloff",
    "balanced interior key and fill lighting with natural skin tones",
]

STUDIO_LIGHTINGS = [
    "balanced studio three-point lighting with realistic skin tones",
    "a large diffused softbox with gentle facial shadows",
    "a controlled key light and subtle rim light against the backdrop",
]

VIEWPOINTS = [
    "at eye level",
    "from a gentle three-quarter angle",
    "from a natural side profile",
    "from a slightly low camera position",
    "from a slightly elevated camera position",
]

FRAMINGS = {
    "face_closeup": {
        "shots": [
            "tight head-and-shoulders portrait",
            "close portrait including the face, neck, shoulders, and one visible hand",
            "three-quarter facial close-up with one hand resting naturally near the chin",
        ],
        "actions": [
            "lightly holding the temple of a pair of eyeglasses",
            "resting one hand against the side of the face",
            "holding a small ceramic cup below the chin",
            "adjusting a scarf near the collar with one hand",
            "gently touching one earring",
        ],
        "lenses": ["an 85mm portrait lens at f/2", "a 50mm lens at f/2.8"],
        "environments": [
            "a quiet urban sidewalk with soft background depth",
            "a bright neighborhood cafe",
            "a green public park",
            "a modern living room with neutral furnishings",
            "an artist workshop with textured surfaces",
            "a library reading area",
            "a simple photography studio with a neutral backdrop",
        ],
        "composition": "The face and visible hand are in sharp focus, with natural facial symmetry and clearly articulated visible fingers",
    },
    "upper_body": {
        "shots": [
            "waist-up environmental portrait with both hands visible",
            "medium portrait from the hips upward",
            "seated upper-body portrait with forearms and hands fully in frame",
        ],
        "actions": [
            "holding an open book with both hands",
            "pouring water from a small pitcher into a glass",
            "typing naturally on a laptop keyboard",
            "folding both arms loosely across the torso",
            "holding a reusable bottle with one hand while the other rests on a table",
            "buttoning the cuff of one sleeve",
            "carrying a small potted plant with both hands",
        ],
        "lenses": ["a 50mm standard lens at f/2.8", "an 85mm portrait lens at f/2.8", "a 35mm lens at f/4"],
        "environments": [
            "a bright neighborhood cafe with tables in the background",
            "a modern living room with neutral furnishings",
            "an artist workshop with shelves and tools",
            "a library reading area with bookshelves",
            "a simple photography studio with a neutral backdrop",
            "a residential kitchen with everyday objects",
            "a covered outdoor terrace",
        ],
        "composition": "Both shoulders, elbows, wrists, and hands remain visible with coherent joints and natural object contact",
    },
    "full_body": {
        "shots": [
            "uncropped full-body photograph from head to toe",
            "vertical full-length environmental portrait",
            "wide full-body composition with space around every limb",
        ],
        "actions": [
            "standing with weight naturally shifted onto one leg and both arms relaxed",
            "walking toward the camera with an alternating arm swing",
            "standing while holding a closed umbrella beside the body",
            "carrying a canvas tote bag in one hand with the other hand visible",
            "standing beside a bicycle with one hand on the handlebar",
            "taking a natural step up a shallow stair",
            "reaching one arm toward a shelf while keeping both feet planted",
        ],
        "lenses": ["a 35mm documentary lens at f/4", "a 50mm standard lens at f/4", "a 24mm lens with restrained perspective distortion"],
        "environments": [
            "a quiet urban sidewalk with realistic street depth",
            "a green public park with a walking path",
            "an outdoor athletics field with visible ground texture",
            "a covered transit platform with architectural leading lines",
            "a simple photography studio with a neutral backdrop",
            "a spacious modern living room",
            "an artist workshop with an open floor area",
        ],
        "composition": "The entire body, both hands, and both feet are visible without cropped limbs, with balanced proportions and credible ground contact",
    },
    "seated_crouching": {
        "shots": [
            "full-body seated portrait with all limbs visible",
            "wide environmental portrait showing a compact crouching pose",
            "three-quarter body photograph with the chair and floor visible",
        ],
        "actions": [
            "sitting upright on a wooden chair with both hands resting separately on the knees",
            "sitting cross-legged on a low bench with relaxed shoulders",
            "crouching to tie one shoelace with both hands clearly interacting with the lace",
            "kneeling on one knee while arranging objects on a low shelf",
            "sitting on the edge of a bench while holding a folded map with both hands",
        ],
        "lenses": ["a 35mm documentary lens at f/4", "a 50mm standard lens at f/4"],
        "environments": [
            "a bright neighborhood cafe with chairs and tables",
            "a green public park with benches and a walking path",
            "a modern living room with a low shelf and neutral furnishings",
            "an artist workshop with a chair and low storage shelves",
            "a library reading area with benches",
            "a simple photography studio with a chair and neutral backdrop",
        ],
        "composition": "The bent knees, elbows, wrists, hands, and feet have plausible articulation and remain clearly separated",
    },
    "dynamic_motion": {
        "shots": [
            "motion-frozen full-body action photograph",
            "wide dynamic photograph with the entire figure visible",
            "full-length side-view action photograph",
        ],
        "actions": [
            "walking briskly across the frame with a natural opposing arm swing",
            "jogging at an easy pace with one foot contacting the ground",
            "throwing a lightweight ball with a clear shoulder-to-wrist motion",
            "stepping sideways while reaching both arms for balance",
            "lifting a small box from a waist-high table using both hands",
            "turning the torso while looking back over one shoulder",
            "stretching both arms overhead with open relaxed hands",
        ],
        "lenses": ["a 35mm lens at 1/1000 second", "a 50mm lens at 1/800 second", "a 24mm lens at 1/1000 second with restrained distortion"],
        "environments": [
            "a broad urban sidewalk with an unobstructed path",
            "a green public park with an open walking path",
            "an outdoor athletics field with visible ground texture",
            "a simple photography studio with a wide neutral backdrop",
            "a spacious artist workshop with a clear floor area",
        ],
        "composition": "The complete body stays in frame with readable limb separation, realistic balance, and physically plausible motion",
    },
    "adult_swimwear": {
        "adult_only": True,
        "shots": [
            "confident full-body swimwear fashion portrait",
            "tasteful glamorous beachwear editorial photograph",
            "full-length poolside fashion photograph",
        ],
        "actions": [
            "standing naturally at the shoreline with both arms relaxed",
            "walking beside the pool with a balanced stride and natural arm swing",
            "sitting upright on the edge of a pool with both hands resting separately beside the hips",
            "standing in a relaxed contrapposto pose with one hand lightly adjusting a sun hat",
            "holding a lightweight beach wrap in both hands with clear finger separation",
        ],
        "outfits_by_sex": {
            "female": [
                "a stylish two-piece bikini with clean fabric edges and a lightweight beach wrap",
                "a sporty two-piece swimsuit with a linen cover-up worn open",
                "a high-waisted bikini with a wide-brimmed sun hat",
            ],
            "male": [
                "tailored swim trunks with an open lightweight linen shirt",
                "sporty fitted swim shorts with a casual beach shirt",
                "classic swim trunks with a lightweight towel draped over one shoulder",
            ],
        },
        "lenses": ["a 50mm standard lens at f/4", "an 85mm portrait lens at f/2.8", "a 35mm editorial lens at f/4"],
        "environments": [
            "a quiet sandy beach with a clear horizon",
            "a modern outdoor swimming pool with uncluttered architecture",
            "a shaded poolside terrace with neutral stone surfaces",
            "a calm tropical shoreline with a clear horizon",
        ],
        "composition": "The adult subject is shown non-explicitly with realistic torso, shoulder, hip, arm, hand, leg, and foot proportions and natural fabric contact",
    },
}


def balanced_values(values, count, rng):
    result = []
    while len(result) < count:
        batch = list(values)
        rng.shuffle(batch)
        result.extend(batch)
    return result[:count]


def subject_for(ethnicity, age_group, sex, rng, minimum_age=None):
    ages = [age for age in AGE_GROUPS[age_group] if minimum_age is None or age >= minimum_age]
    age = rng.choice(ages)
    if age < 18:
        noun = "girl" if sex == "female" else "boy"
    else:
        noun = "woman" if sex == "female" else "man"
    hair_options = HAIR_DETAILS + OLDER_HAIR_DETAILS if age >= 65 else HAIR_DETAILS
    hair = rng.choice(hair_options)
    return f"a {age}-year-old {ethnicity} {noun} {hair}", age


def build_prompt(ethnicity, age_group, sex, framing_name, rng):
    framing = FRAMINGS[framing_name]
    subject, age = subject_for(
        ethnicity,
        age_group,
        sex,
        rng,
        minimum_age=21 if framing.get("adult_only") else None,
    )
    outfit_group = "child" if age < 18 else "adult"
    outfit = rng.choice(framing["outfits_by_sex"][sex]) if "outfits_by_sex" in framing else rng.choice(OUTFITS[outfit_group])
    environment = rng.choice(framing["environments"])
    if "studio" in environment:
        lighting = rng.choice(STUDIO_LIGHTINGS)
    elif any(
        marker in environment
        for marker in ("outdoor", "sidewalk", "park", "field", "platform", "terrace", "beach", "pool", "shoreline")
    ):
        lighting = rng.choice(OUTDOOR_LIGHTINGS)
    else:
        lighting = rng.choice(INDOOR_LIGHTINGS)
    axes = {
        "ethnicity": ethnicity,
        "age_group": age_group,
        "age": age,
        "sex": sex,
        "framing": framing_name,
        "shot": rng.choice(framing["shots"]),
        "action": rng.choice(framing["actions"]),
        "outfit": outfit,
        "expression": rng.choice(EXPRESSIONS),
        "lighting": lighting,
        "environment": environment,
        "viewpoint": rng.choice(VIEWPOINTS),
        "lens": rng.choice(framing["lenses"]),
    }
    prompt = (
        f"A photorealistic {axes['shot']} of {subject}, {axes['action']}, {axes['expression']}. "
        f"The subject is wearing {axes['outfit']} in {axes['environment']}. "
        f"Illuminated by {axes['lighting']}, photographed {axes['viewpoint']} with {axes['lens']}. "
        f"{framing['composition']}. Natural skin texture, realistic fabric response, coherent human anatomy, "
        "documentary-level photographic detail."
    )
    return prompt, axes


def generate_prompts(count, seed):
    rng = random.Random(seed)
    ethnicities = balanced_values(ETHNICITIES, count, rng)
    age_groups = balanced_values(list(AGE_GROUPS), count, rng)
    sexes = balanced_values(SEXES, count, rng)
    framings = balanced_values(list(FRAMINGS), count, rng)
    records = []
    seen = set()
    for ethnicity, age_group, sex, framing in zip(ethnicities, age_groups, sexes, framings):
        if FRAMINGS[framing].get("adult_only") and age_group in {"young_child", "older_child"}:
            framing = rng.choice([name for name, value in FRAMINGS.items() if not value.get("adult_only")])
        while True:
            prompt, axes = build_prompt(ethnicity, age_group, sex, framing, rng)
            if prompt not in seen:
                break
        seen.add(prompt)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        records.append({"prompt_id": f"person-{prompt_hash[:16]}", "prompt": prompt, "axes": axes})
    return records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.output is None:
        args.output = Path(f"my/flux2_distillation_person_prompts_{args.count}.txt")
    return args


def main():
    args = parse_args()
    records = generate_prompts(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{record['prompt']}\n" for record in records), encoding="utf-8")
    metadata_path = args.output.with_suffix(".jsonl")
    metadata_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Saved {len(records):,} unique prompts to {args.output}")
    print(f"Saved prompt metadata to {metadata_path}")


if __name__ == "__main__":
    main()
