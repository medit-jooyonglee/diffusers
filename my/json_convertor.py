import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", default="avngrsstyle superhero")
    return parser.parse_args()


def add_prefix(text, prefix):
    text = text.strip()
    if text == prefix or text.startswith(f"{prefix},"):
        return text
    return f"{prefix}, {text}" if text else prefix


def main():
    args = parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_converted{args.input.suffix}")
    converted = 0
    rows = []

    with args.input.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if "text" not in row or not isinstance(row["text"], str):
                raise ValueError(f"Line {line_number} must contain a string text field")

            updated_text = add_prefix(row["text"], args.prefix)
            converted += updated_text != row["text"]
            row["text"] = updated_text
            rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {output} ({converted} updated)")


if __name__ == "__main__":
    main()
