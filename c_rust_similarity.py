#!/usr/bin/env python3
"""Compute C/Rust embedding similarity from JSONL prompt/completion data."""
import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


C_BLOCK_RE = re.compile(r"\{\{([^}]+\.(?:c|h))\}\}\s*```c\n(.*?)```", re.S)
RUST_BLOCK_RE = re.compile(r"\{\{([a-zA-Z0-9_\-\.]+\.rs)\}\}\s*```rust\s*\n(.*?)```", re.S)
RUST_LAST_BLOCK_RE = re.compile(
    r"\{\{([a-zA-Z0-9_\-\.]+\.rs)\}\}\s*```rust\s*\n((?:(?!```).)*?)$",
    re.S,
)


def _extract_section(text: str, start_marker: str, end_marker: Optional[str]) -> str:
    """Return substring between markers or empty string if start marker missing."""
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    if end_marker:
        end = text.find(end_marker, start)
        if end == -1:
            end = len(text)
    else:
        end = len(text)
    return text[start:end]


def _last_final_solution_block(text: str) -> str:
    """Return the last <final_solution> split segment (PZ04-compatible)."""
    return text.split("<final_solution>")[-1]


def _last_token_pool(last_hidden_states, attention_mask):
    """Pool the last token embedding with left-padding awareness."""
    import torch

    left_padding = attention_mask[:, -1].sum().item() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class QwenEmbedder:
    """Embed texts with Qwen3-Embedding models using Transformers on GPU."""
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 32768,
        batch_size: int = 4,
        require_gpu: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for QwenEmbedder") from exc

        if require_gpu:
            if not device.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError("GPU is required but not available")

        if device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float16
        else:
            model_dtype = torch.float32

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=model_dtype)
        self.model.to(device)
        self.model.eval()

    def encode(self, texts: List[str]):
        """Return L2-normalized embeddings for a list of input texts."""
        torch = self.torch
        import torch.nn.functional as F

        all_embeds = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeds = _last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                embeds = F.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.detach().cpu())
        return torch.cat(all_embeds, dim=0)


@dataclass
class CodePair:
    """Paired C/Rust code payload extracted from a single JSONL line."""
    idx: Optional[int]
    c_path: Optional[str]
    rust_file: str
    c_files: List[str]
    c_code: str
    rust_code: str
    rust_sig_code: Optional[str]
    error: Optional[str]


@dataclass
class SimilarityStats:
    """Running statistics for similarity values."""
    count: int = 0
    total: float = 0.0
    min_val: float = field(default_factory=lambda: float("inf"))
    max_val: float = field(default_factory=lambda: float("-inf"))

    def update(self, value: float) -> None:
        """Update running count, sum, min, and max."""
        self.count += 1
        self.total += value
        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value

    def to_dict(self) -> Dict[str, float]:
        """Return a compact stats dictionary."""
        if self.count == 0:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "count": self.count,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.total / self.count,
        }


@dataclass
class SimilarityBundle:
    """Grouped statistics for code, signature, and delta similarities."""
    code: SimilarityStats = field(default_factory=SimilarityStats)
    sig: SimilarityStats = field(default_factory=SimilarityStats)
    delta: SimilarityStats = field(default_factory=SimilarityStats)

    def update(self, code: float, sig: float, delta: float) -> None:
        """Update all sub-stats at once."""
        self.code.update(code)
        self.sig.update(sig)
        self.delta.update(delta)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Return stats for each cosine field."""
        return {
            "cosine_similarity_code": self.code.to_dict(),
            "cosine_similarity_sig": self.sig.to_dict(),
            "cosine_similarity(code-sig)": self.delta.to_dict(),
        }


class CRustSimilarity:
    """Parse JSONL inputs, build C/Rust pairs, embed, and write similarity results."""
    def __init__(
        self,
        model_path: str = "Qwen3-Embedding-4B",
        device: str = "cuda",
        max_length: int = 32768,
        batch_size: int = 4,
        require_gpu: bool = True,
        embedder=None,
    ) -> None:
        self.embedder = embedder or QwenEmbedder(
            model_path=model_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            require_gpu=require_gpu,
        )

    def parse_jsonl_line(self, line: str) -> List[CodePair]:
        """Parse one JSONL line into one or more C/Rust code pairs."""
        obj = json.loads(line)
        prompt = obj.get("prompt", "")
        completion = obj.get("completion", "")
        if not prompt or not completion:
            raise ValueError("missing prompt or completion")

        c_map = self._parse_c_files(prompt)
        rust_sig_map, sig_error = self._parse_rust_sig_files(prompt)
        rust_map = self._parse_completion_rust(completion)
        assert c_map, "no C files parsed"
        assert rust_map, "no Rust files parsed"

        idx = obj.get("idx")
        c_path = obj.get("c_path")
        return self._build_pairs(
            c_map, rust_map, rust_sig_map, sig_error, idx=idx, c_path=c_path
        )

    def parse_prompt_completion(self, prompt: str, completion: str) -> List[CodePair]:
        """Parse prompt/completion strings into C/Rust code pairs."""
        if not prompt or not completion:
            raise ValueError("missing prompt or completion")
        c_map = self._parse_c_files(prompt)
        rust_sig_map, sig_error = self._parse_rust_sig_files(prompt)
        rust_map = self._parse_completion_rust(completion)
        assert c_map, "no C files parsed"
        assert rust_map, "no Rust files parsed"
        return self._build_pairs(
            c_map, rust_map, rust_sig_map, sig_error, idx=None, c_path=None
        )

    def _parse_c_files(self, prompt: str) -> Dict[str, str]:
        """Extract C code blocks from the prompt section."""
        section = _extract_section(prompt, "The C Source Files", "The Rust Interface Files")
        if not section:
            raise ValueError("C section not found in prompt")
        files = {name: code.strip() for name, code in C_BLOCK_RE.findall(section)}
        if not files:
            raise ValueError("no C files found in prompt")
        return files

    def _parse_rust_sig_files(self, prompt: str) -> tuple[Dict[str, str], Optional[str]]:
        """Extract Rust signature blocks from the prompt section."""
        section = _extract_section(prompt, "The Rust Interface Files", None)
        if not section:
            return {}, "sig_missing_section"
        files = {name: code.strip() for name, code in RUST_BLOCK_RE.findall(section)}
        files = {name: code for name, code in files.items() if code}
        if not files:
            return {}, "sig_missing_blocks"
        return files, None

    def _parse_completion_rust(self, completion: str) -> Dict[str, str]:
        """Extract Rust code blocks from the completion final_solution section."""
        final_block = _last_final_solution_block(completion)
        files = {name: code.strip() for name, code in RUST_BLOCK_RE.findall(final_block)}

        last_match = RUST_LAST_BLOCK_RE.search(final_block)
        if last_match:
            name, code = last_match.groups()
            if name not in files:
                files[name] = code.strip()

        if not files:
            raise ValueError("no Rust files found in final_solution block")
        return files

    def _build_pairs(
        self,
        c_map: Dict[str, str],
        rust_map: Dict[str, str],
        rust_sig_map: Dict[str, str],
        sig_error: Optional[str],
        idx: Optional[int],
        c_path: Optional[str],
    ) -> List[CodePair]:
        """Build matched C/Rust pairs from parsed file maps."""
        pairs: List[CodePair] = []
        if sig_error is None:
            missing = [name for name in rust_map if name not in rust_sig_map]
            if missing:
                c_files = sorted(c_map.keys())
                rust_files = sorted(rust_map.keys())
                sig_files = sorted(rust_sig_map.keys())
                c_code = "\n\n".join(c_map[name] for name in c_files).strip()
                rust_code = "\n\n".join(rust_map[name] for name in rust_files).strip()
                rust_sig_code = "\n\n".join(rust_sig_map[name] for name in sig_files).strip()
                if not c_code:
                    raise ValueError("empty merged C for fallback")
                if not rust_code:
                    raise ValueError("empty merged Rust for fallback")
                if not rust_sig_code:
                    rust_sig_code = None
                pairs.append(
                    CodePair(
                        idx=idx,
                        c_path=c_path,
                        rust_file="__all__",
                        c_files=c_files,
                        c_code=c_code,
                        rust_code=rust_code,
                        rust_sig_code=rust_sig_code,
                        error="sig_mismatch_fallback_all",
                    )
                )
                return pairs

        for rust_file, rust_code in rust_map.items():
            base = os.path.splitext(rust_file)[0]
            c_files = []
            parts = []
            header = base + ".h"
            source = base + ".c"
            if header in c_map:
                c_files.append(header)
                parts.append(c_map[header])
            if source in c_map:
                c_files.append(source)
                parts.append(c_map[source])
            if not parts:
                raise ValueError(f"missing C files for rust base {base}")
            merged = "\n\n".join(parts).strip()
            if not merged:
                raise ValueError(f"empty merged C for rust base {base}")
            rust_sig_code = rust_sig_map.get(rust_file) if sig_error is None else None
            if rust_sig_code:
                rust_sig_code = rust_sig_code.strip()
                if not rust_sig_code:
                    rust_sig_code = None
            pairs.append(
                CodePair(
                    idx=idx,
                    c_path=c_path,
                    rust_file=rust_file,
                    c_files=c_files,
                    c_code=merged,
                    rust_code=rust_code.strip(),
                    rust_sig_code=rust_sig_code,
                    error=sig_error,
                )
            )
        return pairs

    def _cosine_similarity(self, c_embeds, r_embeds) -> List[float]:
        """Compute cosine similarity for aligned embedding rows."""
        torch = self.embedder.torch
        assert c_embeds.shape == r_embeds.shape
        sims = torch.sum(c_embeds * r_embeds, dim=1)
        return sims.tolist()

    def _compute_scores(
        self, pairs: List[CodePair]
    ) -> List[tuple[float, float, float, Optional[str]]]:
        """Compute code/sig/delta cosine scores with missing/negative handling."""
        if not pairs:
            return []
        results: List[tuple[float, float, float, Optional[str]]] = [
            (0.0, 0.0, 0.0, pair.error) for pair in pairs
        ]
        idxs = [i for i, pair in enumerate(pairs) if pair.rust_sig_code]
        idx_set = set(idxs)
        for i, pair in enumerate(pairs):
            if i in idx_set:
                continue
            if results[i][3] is None:
                results[i] = (0.0, 0.0, 0.0, "sig_missing")
        if not idxs:
            return results

        c_texts = [pairs[i].c_code for i in idxs]
        r_texts = [pairs[i].rust_code for i in idxs]
        sig_texts = [pairs[i].rust_sig_code or "" for i in idxs]
        c_embeds = self.embedder.encode(c_texts)
        r_embeds = self.embedder.encode(r_texts)
        sig_embeds = self.embedder.encode(sig_texts)
        code_sims = self._cosine_similarity(c_embeds, r_embeds)
        sig_sims = self._cosine_similarity(c_embeds, sig_embeds)

        for idx, code_sim, sig_sim in zip(idxs, code_sims, sig_sims):
            code_sim = float(code_sim)
            sig_sim = float(sig_sim)
            delta = code_sim - sig_sim
            error = pairs[idx].error
            if delta < 0:
                if error:
                    error = f"{error}|delta_negative"
                else:
                    error = "delta_negative"
                results[idx] = (0.0, 0.0, 0.0, error)
            else:
                results[idx] = (code_sim, sig_sim, delta, error)
        return results

    def score_prompt_completion(self, prompt: str, completion: str) -> List[Dict[str, object]]:
        """Score a single prompt/completion and return per-pair similarity results."""
        pairs = self.parse_prompt_completion(prompt, completion)
        return self._score_pairs(pairs)

    def score_texts(
        self, c_code: str, rust_code: str, rust_sig_code: Optional[str] = None
    ) -> Dict[str, float]:
        """Score a single C/Rust pair with optional Rust signature code."""
        if not c_code or not rust_code:
            raise ValueError("missing c_code or rust_code")
        pair = CodePair(
            idx=None,
            c_path=None,
            rust_file="inline.rs",
            c_files=[],
            c_code=c_code,
            rust_code=rust_code,
            rust_sig_code=rust_sig_code.strip() if rust_sig_code else None,
            error=None,
        )
        code_sim, sig_sim, delta, error = self._compute_scores([pair])[0]
        return {
            "cosine_similarity_code": code_sim,
            "cosine_similarity_sig": sig_sim,
            "cosine_similarity(code-sig)": delta,
            "error": error,
        }

    def process_file(
        self,
        input_path: str,
        output_path: str,
        limit: Optional[int] = None,
        fail_fast: bool = False,
        error_cap: int = 20,
    ) -> Dict[str, object]:
        """Process an input JSONL file and write similarity JSONL output."""
        stats = SimilarityBundle()
        errors = []
        pairs_written = 0
        lines_processed = 0
        buffer_pairs: List[CodePair] = []

        with open(input_path, "r", encoding="utf-8") as src, open(
            output_path, "w", encoding="utf-8"
        ) as out:
            for line_idx, line in enumerate(src):
                if limit is not None and line_idx >= limit:
                    break
                lines_processed += 1
                try:
                    pairs = self.parse_jsonl_line(line)
                except Exception as exc:
                    if fail_fast:
                        raise
                    if len(errors) < error_cap:
                        errors.append({"line": line_idx, "error": str(exc)})
                    continue
                buffer_pairs.extend(pairs)
                if len(buffer_pairs) >= self.embedder.batch_size:
                    pairs_written += self._write_pairs(buffer_pairs, out, stats)
                    buffer_pairs = []

            if buffer_pairs:
                pairs_written += self._write_pairs(buffer_pairs, out, stats)

        return {
            "input_path": input_path,
            "output_path": output_path,
            "lines_processed": lines_processed,
            "pairs_written": pairs_written,
            "errors": errors,
            "similarity": stats.to_dict(),
        }

    def _write_pairs(self, pairs: List[CodePair], out, stats: SimilarityBundle) -> int:
        """Embed and write a batch of pairs, updating stats."""
        if not pairs:
            return 0
        start = time.perf_counter()
        scores = self._compute_scores(pairs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_pair_ms = elapsed_ms / len(pairs)
        for pair, (code_sim, sig_sim, delta, error) in zip(pairs, scores):
            record = {
                "cosine_similarity_code": code_sim,
                "cosine_similarity_sig": sig_sim,
                "cosine_similarity(code-sig)": delta,
                "elapsed_ms": per_pair_ms,
                "error": error,
                "idx": pair.idx,
                "c_path": pair.c_path,
                "rust_file": pair.rust_file,
                "c_files": pair.c_files,
                "c_code": pair.c_code,
                "rust_code": pair.rust_code,
                "rust_sig": pair.rust_sig_code or "",
            }
            json.dump(record, out, ensure_ascii=True)
            out.write("\n")
            stats.update(code_sim, sig_sim, delta)
        return len(pairs)

    def _score_pairs(self, pairs: List[CodePair]) -> List[Dict[str, object]]:
        """Score pairs in-memory and return minimal results."""
        if not pairs:
            return []
        scores = self._compute_scores(pairs)
        results = []
        for pair, (code_sim, sig_sim, delta, error) in zip(pairs, scores):
            results.append(
                {
                    "rust_file": pair.rust_file,
                    "c_files": pair.c_files,
                    "cosine_similarity_code": code_sim,
                    "cosine_similarity_sig": sig_sim,
                    "cosine_similarity(code-sig)": delta,
                    "error": error,
                }
            )
        return results


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute C/Rust similarity from train_outputs JSONL.")
    parser.add_argument("--input", default="train_outputs_0.jsonl", help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model-path", default="Qwen3-Embedding-4B", help="Local model path")
    parser.add_argument("--device", default="cuda", help="Device, default cuda")
    parser.add_argument("--limit", type=int, help="Process only first N lines")
    parser.add_argument("--batch-size", type=int, default=4, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=32768, help="Tokenizer max length")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first parse error")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU if no GPU")
    args = parser.parse_args()

    pipeline = CRustSimilarity(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        require_gpu=not args.allow_cpu,
    )
    report = pipeline.process_file(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
