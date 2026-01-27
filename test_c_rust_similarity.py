import json
import tempfile
import unittest

from c_rust_similarity import CRustSimilarity


class DummyEmbedder:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.batch_size = 4

    def encode(self, texts):
        return self.torch.tensor([[1.0, 0.0]] * len(texts), dtype=self.torch.float32)


def _sample_jsonl_line() -> str:
    prompt = (
        "The C Source Files\n"
        "{{foo.h}}\n"
        "```c\n"
        "/* header */\n"
        "int foo(void);\n"
        "```\n"
        "\n"
        "{{foo.c}}\n"
        "```c\n"
        "int foo(void) { return 1; }\n"
        "```\n"
        "\n"
        "The Rust Interface Files are:\n"
        "{{foo.rs}}\n"
        "```rust\n"
        "pub fn foo() -> i32 { unimplemented!() }\n"
        "```\n"
        "Return your final solution\n"
    )
    completion = (
        "intro <final_solution>{{bad.rs}}\n"
        "```rust\n"
        "bad\n"
        "```\n"
        "</final_solution>\n"
        "middle <final_solution>\n"
        "{{foo.rs}}\n"
        "```rust\n"
        "pub fn foo() -> i32 { 1 }\n"
        "```\n"
        "</final_solution>\n"
    )
    obj = {"idx": 1, "c_path": "path", "prompt": prompt, "completion": completion}
    return json.dumps(obj, ensure_ascii=True)


class CRustSimilarityTests(unittest.TestCase):
    def test_parse_jsonl_line(self):
        pipeline = CRustSimilarity(embedder=DummyEmbedder())
        pairs = pipeline.parse_jsonl_line(_sample_jsonl_line())
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair.rust_file, "foo.rs")
        self.assertEqual(pair.c_files, ["foo.h", "foo.c"])
        self.assertTrue(pair.c_code.startswith("/* header */"))
        self.assertIn("int foo(void)", pair.c_code)
        self.assertIn("pub fn foo()", pair.rust_code)
        self.assertIn("pub fn foo()", pair.rust_sig_code or "")

    def test_process_file(self):
        pipeline = CRustSimilarity(embedder=DummyEmbedder())
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = f"{tmpdir}/in.jsonl"
            out_path = f"{tmpdir}/out.jsonl"
            with open(in_path, "w", encoding="utf-8") as f:
                f.write(_sample_jsonl_line() + "\n")
            report = pipeline.process_file(in_path, out_path, limit=1)
            self.assertEqual(report["pairs_written"], 1)
            with open(out_path, "r", encoding="utf-8") as f:
                rec = json.loads(f.readline())
            self.assertAlmostEqual(rec["cosine_similarity_code"], 1.0, places=6)
            self.assertAlmostEqual(rec["cosine_similarity_sig"], 1.0, places=6)
            self.assertAlmostEqual(rec["cosine_similarity(code-sig)"], 0.0, places=6)

    def test_score_prompt_completion(self):
        pipeline = CRustSimilarity(embedder=DummyEmbedder())
        obj = json.loads(_sample_jsonl_line())
        results = pipeline.score_prompt_completion(obj["prompt"], obj["completion"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rust_file"], "foo.rs")
        self.assertAlmostEqual(results[0]["cosine_similarity_code"], 1.0, places=6)
        self.assertAlmostEqual(results[0]["cosine_similarity_sig"], 1.0, places=6)
        self.assertAlmostEqual(results[0]["cosine_similarity(code-sig)"], 0.0, places=6)

    def test_score_texts(self):
        pipeline = CRustSimilarity(embedder=DummyEmbedder())
        result = pipeline.score_texts(
            "int foo(void) { return 1; }",
            "pub fn foo() -> i32 { 1 }",
            "pub fn foo() -> i32;",
        )
        self.assertAlmostEqual(result["cosine_similarity_code"], 1.0, places=6)
        self.assertAlmostEqual(result["cosine_similarity_sig"], 1.0, places=6)
        self.assertAlmostEqual(result["cosine_similarity(code-sig)"], 0.0, places=6)

    def test_score_texts_missing_sig(self):
        pipeline = CRustSimilarity(embedder=DummyEmbedder())
        result = pipeline.score_texts("int foo(void) { return 1; }", "pub fn foo() -> i32 { 1 }")
        self.assertAlmostEqual(result["cosine_similarity_code"], 0.0, places=6)
        self.assertAlmostEqual(result["cosine_similarity_sig"], 0.0, places=6)
        self.assertAlmostEqual(result["cosine_similarity(code-sig)"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
