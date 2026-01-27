#!/usr/bin/env bash
set -euo pipefail

# Small concurrent check for /score_texts. Requires a running server.
BASE_URL="${BASE_URL:-http://127.0.0.1:7010}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-10}"
CONCURRENCY="${CONCURRENCY:-5}"

export BASE_URL
export PAYLOAD='{"c_code":"int foo(void){return 1;}","rust_code":"pub fn foo()->i32{1}","rust_sig_code":"pub fn foo()->i32;"}'

echo "BASE_URL=${BASE_URL}"
echo "TOTAL_REQUESTS=${TOTAL_REQUESTS}"
echo "CONCURRENCY=${CONCURRENCY}"

# Use -n 1 to avoid placeholder substitution inside variable names.
responses=$(seq 1 "${TOTAL_REQUESTS}" | xargs -P "${CONCURRENCY}" -n 1 sh -c \
  'curl -s "$BASE_URL/score_texts" -H "Content-Type: application/json" -d "$PAYLOAD"')

count=$(printf "%s" "${responses}" | python -c 'import sys; data=sys.stdin.read(); print(data.count("cosine_similarity_code"))')
echo "responses=${count}/${TOTAL_REQUESTS}"
