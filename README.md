# C/Rust Similarity Pipeline
[English](#english) | [中文](#中文)

## English
### Why This
- Pair C headers/sources with Rust outputs per file and score semantic closeness.
- GPU embedding with Qwen3-Embedding-4B; output code/sig/delta cosine scores.
- Client/server separation: the server loads the model once; requests are queued and dynamically batched for high concurrency.

### Usage (3 Ways)
#### 1) HTTP Service (Python client)
Start the service (from repo root):
```bash
python api_server.py --model-path ../Qwen3-Embedding-4B --host 0.0.0.0 --port 7010
```

Python client:
```python
import json
from urllib import request

payload = {"prompt": "...", "completion": "..."}
data = json.dumps(payload).encode("utf-8")
req = request.Request(
    "http://127.0.0.1:7010/score_prompt_completion",
    data=data,
    headers={"Content-Type": "application/json"},
)
with request.urlopen(req) as resp:
    print(resp.read().decode("utf-8"))
```

#### 2) Quick API (in-process)
```python
from c_rust_similarity import CRustSimilarity

pipe = CRustSimilarity(model_path="../Qwen3-Embedding-4B", device="cuda")
print(pipe.score_prompt_completion(prompt, completion))
print(pipe.score_texts(c_code, rust_code, rust_sig_code))
```

#### 3) Direct File Processing
```bash
python c_rust_similarity.py \
  --input ../c_rust_sample/train_outputs_0.jsonl \
  --output ../c_rust_sample/c_rust_similarity.jsonl \
  --model-path ../Qwen3-Embedding-4B \
  --max-length 32768 \
  --batch-size 1
```

## 中文
### 优点 / 为什么
- 按文件匹配 C 头文件/源码与 Rust 输出，衡量语义接近度。
- 使用 Qwen3-Embedding-4B 的 GPU 向量，输出 code/sig/delta 余弦分数。
- C/S 分离：服务端一次加载模型，请求入队并动态 batch，支撑高并发。

### 使用方式（3 种）
#### 1) HTTP 服务（Python 客户端）
启动服务（仓库根目录）：
```bash
python api_server.py --model-path ../Qwen3-Embedding-4B --host 0.0.0.0 --port 7010
```

Python 客户端：
```python
import json
from urllib import request

payload = {"prompt": "...", "completion": "..."}
data = json.dumps(payload).encode("utf-8")
req = request.Request(
    "http://127.0.0.1:7010/score_prompt_completion",
    data=data,
    headers={"Content-Type": "application/json"},
)
with request.urlopen(req) as resp:
    print(resp.read().decode("utf-8"))
```

#### 2) API 快捷调用（进程内）
```python
from c_rust_similarity import CRustSimilarity

pipe = CRustSimilarity(model_path="../Qwen3-Embedding-4B", device="cuda")
print(pipe.score_prompt_completion(prompt, completion))
print(pipe.score_texts(c_code, rust_code, rust_sig_code))
```

#### 3) 直接处理文件
```bash
python c_rust_similarity.py \
  --input ../c_rust_sample/train_outputs_0.jsonl \
  --output ../c_rust_sample/c_rust_similarity.jsonl \
  --model-path ../Qwen3-Embedding-4B \
  --max-length 32768 \
  --batch-size 1
```
