#!/usr/bin/env bash
set -euo pipefail

# Simple functional checks for API endpoints. Requires a running server.
BASE_URL="${BASE_URL:-http://127.0.0.1:7010}"

echo "BASE_URL=${BASE_URL}"

echo "==> score_texts"
curl -s "${BASE_URL}/score_texts" \
  -H "Content-Type: application/json" \
  -d '{"c_code":"int foo(void){return 1;}","rust_code":"pub fn foo()->i32{1}","rust_sig_code":"pub fn foo()->i32;"}'
echo

echo "==> score_prompt_completion"
python - <<'PY' | curl -s "${BASE_URL}/score_prompt_completion" -H "Content-Type: application/json" -d @-
import json
prompt = """The C Source Files
{{foo.h}}```c
int foo(void);
```
{{foo.c}}```c
int foo(void){return 1;}
```
The Rust Interface Files
{{foo.rs}}```rust
pub fn foo() -> i32;
```
"""
completion = """<final_solution>
{{foo.rs}}```rust
pub fn foo() -> i32 { 1 }
```
"""
print(json.dumps({"prompt": prompt, "completion": completion}))
PY
echo

echo "==> score_texts_batch"
python - <<'PY' | curl -s "${BASE_URL}/score_texts_batch" -H "Content-Type: application/json" -d @-
import json
payload = {
    "items": [
        {
            "c_code": "int foo(void){return 1;}",
            "rust_code": "pub fn foo()->i32{1}",
            "rust_sig_code": "pub fn foo()->i32;",
        },
        {
            "c_code": "int add(int a,int b){return a+b;}",
            "rust_code": "pub fn add(a:i32,b:i32)->i32{a+b}",
            "rust_sig_code": "pub fn add(a:i32,b:i32)->i32;",
        },
    ]
}
print(json.dumps(payload))
PY
echo

echo "==> score_prompt_completion_batch"
python - <<'PY' | curl -s "${BASE_URL}/score_prompt_completion_batch" -H "Content-Type: application/json" -d @-
import json
prompt1 = """The C Source Files
{{foo.h}}```c
int foo(void);
```
{{foo.c}}```c
int foo(void){return 1;}
```
The Rust Interface Files
{{foo.rs}}```rust
pub fn foo() -> i32;
```
"""
completion1 = """<final_solution>
{{foo.rs}}```rust
pub fn foo() -> i32 { 1 }
```
"""
prompt2 = """The C Source Files
{{bar.c}}```c
int bar(int x){return x+2;}
```
The Rust Interface Files
{{bar.rs}}```rust
pub fn bar(x: i32) -> i32;
```
"""
completion2 = """<final_solution>
{{bar.rs}}```rust
pub fn bar(x: i32) -> i32 { x + 2 }
```
"""
print(json.dumps({"items": [{"prompt": prompt1, "completion": completion1}, {"prompt": prompt2, "completion": completion2}]}))
PY
echo
