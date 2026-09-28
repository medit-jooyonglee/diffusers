import argparse
import json
import os

from datasets import load_dataset


def save_magicbrush_subset(
    output_dir,
    num_samples=5000,
    split="train",
    shuffle=True,
    seed=42,
):
    ds = load_dataset(
        "osunlp/MagicBrush",
        split=split,
        streaming=True,
    )

    if shuffle:
        ds = ds.shuffle(
            seed=seed,
            buffer_size=2000,
        )

    if num_samples is not None:
        ds = ds.take(num_samples)

    split_dir = os.path.join(output_dir, split)

    source_dir = os.path.join(split_dir, "source")
    target_dir = os.path.join(split_dir, "target")
    mask_dir = os.path.join(split_dir, "mask")

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    metadata_path = os.path.join(
        split_dir,
        "metadata.jsonl"
    )

    with open(metadata_path, "w", encoding="utf-8") as f:

        saved_count = 0
        for idx, sample in enumerate(ds):

            filename = f"{idx:06d}.png"

            source_path = os.path.join(
                source_dir,
                filename,
            )

            target_path = os.path.join(
                target_dir,
                filename,
            )

            mask_path = os.path.join(
                mask_dir,
                filename,
            )

            # -------------------------------------------------
            # save images
            # -------------------------------------------------

            sample["source_img"].save(source_path)
            sample["target_img"].save(target_path)
            sample["mask_img"].save(mask_path)

            # -------------------------------------------------
            # preserve original metadata
            # -------------------------------------------------

            item = {
                "img_id": sample.get("img_id"),
                "turn_index": sample.get("turn_index"),
                "instruction": sample.get("instruction"),

                "source_img": os.path.relpath(
                    source_path,
                    split_dir,
                ),
                "target_img": os.path.relpath(
                    target_path,
                    split_dir,
                ),
                "mask_img": os.path.relpath(
                    mask_path,
                    split_dir,
                ),
            }

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                ) + "\n"
            ) 
            saved_count = idx + 1

            if idx % 100 == 0:
                f.flush()
                print(
                    f"saved {idx}/{num_samples}"
                )
        f.flush()

    print(f"completed: saved {saved_count} samples to {split_dir}")
    if num_samples is not None and saved_count < num_samples:
        print(f"requested {num_samples} samples, but split {split!r} provided {saved_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/data1/jooyonglee/public/magicbrush_subset-20000",
    )
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    save_magicbrush_subset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        split=args.split,
        shuffle=args.shuffle,
        seed=args.seed,
    )
