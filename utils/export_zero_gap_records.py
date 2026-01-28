#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def export_zero_gap(input_path: Path, output_path: Path) -> int:
    count = 0
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("cosine_similarity(code-sig)") == 0:
                f_out.write(json.dumps(obj, ensure_ascii=True))
                f_out.write("\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Export records with zero code-sig gap.")
    parser.add_argument(
        "--input",
        default="../c_rust_sample/c_rust_similarity_code_sig_delta_4b.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="../c_rust_sample/c_rust_similarity_code_sig_delta_4b_zero.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = export_zero_gap(input_path, output_path)
    print(f"written={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
