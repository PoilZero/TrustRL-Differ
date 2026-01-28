#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FIELD_TO_FILENAME = {
    "cosine_similarity_code": "cosine_similarity_code.png",
    "cosine_similarity_sig": "cosine_similarity_sig.png",
    "cosine_similarity(code-sig)": "cosine_similarity_code-sig.png",
    "elapsed_ms": "elapsed_ms.png",
}


def load_series(path: Path):
    fields = list(FIELD_TO_FILENAME.keys())
    series = {field: [] for field in fields}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            code_val = float(obj.get("cosine_similarity_code", 0.0))
            sig_val = float(obj.get("cosine_similarity_sig", 0.0))
            gap_val = code_val - sig_val
            if gap_val < 0:
                gap_val = 0.0
            series["cosine_similarity_code"].append(code_val)
            series["cosine_similarity_sig"].append(sig_val)
            series["cosine_similarity(code-sig)"].append(gap_val)
            series["elapsed_ms"].append(float(obj.get("elapsed_ms", 0.0)))
    return series


def save_line_plot(values, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(values)), values, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("record_index")
    ax.set_ylabel(title)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot similarity series from JSONL output.")
    parser.add_argument(
        "--input",
        default="../c_rust_sample/c_rust_similarity_code_sig_delta_4b.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output-dir",
        default="../c_rust_sample",
        help="Directory to write PNG plots",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series = load_series(input_path)
    for field, values in series.items():
        save_line_plot(values, field, output_dir / FIELD_TO_FILENAME[field])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
