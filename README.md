# C/Rust Similarity Pipeline
[English](#english) | [中文](#中文)

## English
### What It Does
This project parses `train_outputs_0.jsonl`, pairs C source code with Rust outputs, embeds both with `Qwen3-Embedding-0.6B`, computes cosine similarity, and saves results to JSONL with a normalized 0–1 score.

### Structure
- `c_rust_similarity.py`: main pipeline (parsing, pairing, embedding, similarity, output).
- `test_c_rust_similarity.py`: minimal tests with a dummy embedder.
- `utils/`: helper scripts.
- `../Qwen3-Embedding-0.6B/`: local embedding model files (outside repo).
- `../c_rust_sample/`: example input/output location used for full runs and reports.
- `../c_rust_sample/c_rust_similarity.jsonl`: full-run similarity output.
- `../c_rust_sample/c_rust_similarity_report.md`: summary report with statistics.

### Data Flow
1) Read each JSONL line.
2) Parse C files from the prompt section `The C Source Files ... The Rust Interface Files`.
3) Parse Rust files from the **last** `<final_solution> ... </final_solution>` block in the completion.
4) Match Rust `name.rs` to C `name.h` + `name.c`, then merge header + source.
5) Embed C text and Rust text, compute cosine similarity, and map to 0–1.
6) Write one JSONL record per C/Rust pair.

### Similarity Calculation
- Embedding: Qwen3 last-token pooling with left-padding awareness.
- Normalization: L2 normalize embeddings.
- Cosine similarity: dot product of normalized vectors.
- Normalized score: `(sim + 1) / 2`, mapping [-1, 1] to [0, 1].

### Output Schema (JSONL)
Each line contains:
```
{
  "idx": <int|null>,
  "c_path": "<string|null>",
  "rust_file": "<name.rs>",
  "c_files": ["name.h", "name.c"],
  "c_code": "<merged_c_text>",
  "rust_code": "<rust_text>",
  "cosine_similarity": <float>,
  "similarity_0_1": <float>
}
```

### Running
Example run (GPU, full file, from repo root):
```bash
python c_rust_similarity.py \
  --input ../c_rust_sample/train_outputs_0.jsonl \
  --output ../c_rust_sample/c_rust_similarity.jsonl \
  --model-path ../Qwen3-Embedding-0.6B \
  --max-length 32768 \
  --batch-size 1
```

Optional flags:
- `--limit N`: process only first N lines.
- `--allow-cpu`: skip GPU requirement.
- `--fail-fast`: stop on the first parse error.

### Notes
- Default `max_length` is 32768. Longer texts are truncated by the tokenizer.
- Some lines may have no Rust output in `<final_solution>`, which are reported as parse errors.

## 中文
### 功能说明
该项目解析 `train_outputs_0.jsonl`，将 C 源码与 Rust 输出配对，用 `Qwen3-Embedding-0.6B` 生成向量，计算余弦相似度，并保存到 JSONL，同时提供 0–1 归一化分数。

### 目录结构
- `c_rust_similarity.py`：主流程（解析、配对、embedding、相似度、输出）。
- `test_c_rust_similarity.py`：最小化测试（使用 Dummy embedder）。
- `utils/`：辅助脚本。
- `../Qwen3-Embedding-0.6B/`：本地模型文件（仓库外）。
- `../c_rust_sample/`：示例输入/输出目录。
- `../c_rust_sample/c_rust_similarity.jsonl`：全量相似度输出。
- `../c_rust_sample/c_rust_similarity_report.md`：包含统计指标的汇总报告。

### 处理流程
1) 读取每一行 JSONL。
2) 从 prompt 的 `The C Source Files ... The Rust Interface Files` 提取 C 代码块。
3) 从 completion **最后一对** `<final_solution> ... </final_solution>` 提取 Rust 代码块。
4) 通过文件 base 名匹配，将 `name.h` + `name.c` 合并为同一份 C 文本。
5) 生成 C/Rust 向量，计算余弦相似度并映射到 0–1。
6) 每对输出一行 JSONL 记录。

### 相似度计算原理
- 使用 last-token pooling（考虑 left padding）。
- L2 归一化向量。
- 点积得到余弦相似度。
- `(sim + 1) / 2` 将范围映射到 [0, 1]。

### 输出格式（JSONL）
每行示例：
```
{
  "idx": <int|null>,
  "c_path": "<string|null>",
  "rust_file": "<name.rs>",
  "c_files": ["name.h", "name.c"],
  "c_code": "<合并后的C文本>",
  "rust_code": "<Rust文本>",
  "cosine_similarity": <float>,
  "similarity_0_1": <float>
}
```

### 运行方式
GPU 全量示例（在仓库根目录执行）：
```bash
python c_rust_similarity.py \
  --input ../c_rust_sample/train_outputs_0.jsonl \
  --output ../c_rust_sample/c_rust_similarity.jsonl \
  --model-path ../Qwen3-Embedding-0.6B \
  --max-length 32768 \
  --batch-size 1
```

可选参数：
- `--limit N`：只处理前 N 行。
- `--allow-cpu`：允许使用 CPU。
- `--fail-fast`：出现解析错误立即中止。

### 备注
- 默认 `max_length` 为 32768，超长文本会被 tokenizer 截断。
- 部分样本可能在 `<final_solution>` 中没有 Rust 代码，会记录为解析错误。
