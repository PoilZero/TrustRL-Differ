#!/usr/bin/env python3
"""HTTP API for prompt/completion and text similarity scoring."""
import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from c_rust_similarity import CRustSimilarity, CodePair


@dataclass
class BatchItem:
    kind: str
    prompt: Optional[str]
    completion: Optional[str]
    c_code: Optional[str]
    rust_code: Optional[str]
    rust_sig_code: Optional[str]
    request_id: Optional[str]
    future: asyncio.Future


class PromptCompletionRequest(BaseModel):
    prompt: str
    completion: str
    request_id: Optional[str] = None


class PromptCompletionBatchRequest(BaseModel):
    items: List[PromptCompletionRequest]


class TextsRequest(BaseModel):
    c_code: str
    rust_code: str
    rust_sig_code: Optional[str] = None
    request_id: Optional[str] = None


class TextsBatchRequest(BaseModel):
    items: List[TextsRequest]


class BatchScoringService:
    """Queue requests and process them in batches to avoid per-request blocking."""
    def __init__(
        self,
        model_path: str,
        device: str,
        max_length: int,
        max_batch_size: int,
        max_wait_ms: int,
        max_queue_size: int,
        allow_cpu: bool,
    ) -> None:
        self.model = CRustSimilarity(
            model_path=model_path,
            device=device,
            max_length=max_length,
            batch_size=max_batch_size,
            require_gpu=not allow_cpu,
        )
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.queue: asyncio.Queue[BatchItem] = asyncio.Queue(maxsize=max_queue_size)
        self._worker: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def enqueue(self, item: BatchItem) -> Dict[str, Any]:
        if self.queue.full():
            raise RuntimeError("queue full")
        await self.queue.put(item)
        return await item.future

    async def _batch_loop(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            start = time.monotonic()
            while len(batch) < self.max_batch_size:
                remain = self.max_wait_s - (time.monotonic() - start)
                if remain <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remain)
                except asyncio.TimeoutError:
                    break
                batch.append(item)

            results = await asyncio.to_thread(self._process_batch_sync, batch)
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

    def _process_batch_sync(self, batch: List[BatchItem]) -> List[Dict[str, Any]]:
        responses: List[Optional[Dict[str, Any]]] = [None] * len(batch)

        pc_pairs: List[CodePair] = []
        pc_pair_to_item: List[int] = []
        for idx, item in enumerate(batch):
            if item.kind != "prompt_completion":
                continue
            try:
                pairs = self.model.parse_prompt_completion(item.prompt or "", item.completion or "")
            except Exception as exc:
                responses[idx] = {
                    "request_id": item.request_id,
                    "results": [],
                    "error": str(exc),
                }
                continue
            for pair in pairs:
                pc_pairs.append(pair)
                pc_pair_to_item.append(idx)

        pc_results: Dict[int, List[Dict[str, Any]]] = {}
        if pc_pairs:
            scores = self.model._compute_scores(pc_pairs)
            for pair, (code_sim, sig_sim, delta, error), item_idx in zip(
                pc_pairs, scores, pc_pair_to_item
            ):
                pc_results.setdefault(item_idx, []).append(
                    {
                        "rust_file": pair.rust_file,
                        "c_files": pair.c_files,
                        "cosine_similarity_code": code_sim,
                        "cosine_similarity_sig": sig_sim,
                        "cosine_similarity(code-sig)": delta,
                        "error": error,
                    }
                )

        for idx, item in enumerate(batch):
            if item.kind != "prompt_completion":
                continue
            if responses[idx] is not None:
                continue
            responses[idx] = {
                "request_id": item.request_id,
                "results": pc_results.get(idx, []),
                "error": None,
            }

        text_items: List[int] = []
        text_pairs: List[CodePair] = []
        for idx, item in enumerate(batch):
            if item.kind != "texts":
                continue
            if not item.c_code or not item.rust_code:
                responses[idx] = {
                    "request_id": item.request_id,
                    "result": None,
                    "error": "missing c_code or rust_code",
                }
                continue
            text_items.append(idx)
            text_pairs.append(
                CodePair(
                    idx=None,
                    c_path=None,
                    rust_file="inline.rs",
                    c_files=[],
                    c_code=item.c_code,
                    rust_code=item.rust_code,
                    rust_sig_code=item.rust_sig_code.strip() if item.rust_sig_code else None,
                    error=None,
                )
            )

        if text_pairs:
            scores = self.model._compute_scores(text_pairs)
            for item_idx, (code_sim, sig_sim, delta, error) in zip(text_items, scores):
                responses[item_idx] = {
                    "request_id": batch[item_idx].request_id,
                    "result": {
                        "cosine_similarity_code": code_sim,
                        "cosine_similarity_sig": sig_sim,
                        "cosine_similarity(code-sig)": delta,
                        "error": error,
                    },
                    "error": None,
                }

        for i, item in enumerate(batch):
            if responses[i] is None:
                responses[i] = {
                    "request_id": item.request_id,
                    "error": "internal error",
                }

        return [resp for resp in responses]


def create_app(service: BatchScoringService) -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    async def _startup() -> None:
        await service.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await service.stop()

    @app.post("/score_prompt_completion")
    async def score_prompt_completion(req: PromptCompletionRequest) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        item = BatchItem(
            kind="prompt_completion",
            prompt=req.prompt,
            completion=req.completion,
            c_code=None,
            rust_code=None,
            rust_sig_code=None,
            request_id=req.request_id,
            future=loop.create_future(),
        )
        try:
            return await service.enqueue(item)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/score_prompt_completion_batch")
    async def score_prompt_completion_batch(req: PromptCompletionBatchRequest) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        items = [
            BatchItem(
                kind="prompt_completion",
                prompt=item.prompt,
                completion=item.completion,
                c_code=None,
                rust_code=None,
                rust_sig_code=None,
                request_id=item.request_id,
                future=loop.create_future(),
            )
            for item in req.items
        ]
        try:
            results = await asyncio.gather(*(service.enqueue(item) for item in items))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"results": results}

    @app.post("/score_texts")
    async def score_texts(req: TextsRequest) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        item = BatchItem(
            kind="texts",
            prompt=None,
            completion=None,
            c_code=req.c_code,
            rust_code=req.rust_code,
            rust_sig_code=req.rust_sig_code,
            request_id=req.request_id,
            future=loop.create_future(),
        )
        try:
            return await service.enqueue(item)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/score_texts_batch")
    async def score_texts_batch(req: TextsBatchRequest) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        items = [
            BatchItem(
                kind="texts",
                prompt=None,
                completion=None,
                c_code=item.c_code,
                rust_code=item.rust_code,
                rust_sig_code=item.rust_sig_code,
                request_id=item.request_id,
                future=loop.create_future(),
            )
            for item in req.items
        ]
        try:
            results = await asyncio.gather(*(service.enqueue(item) for item in items))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"results": results}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the similarity API service.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=7010, help="Bind port")
    parser.add_argument("--model-path", default="../Qwen3-Embedding-4B", help="Model path")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--max-length", type=int, default=32768, help="Tokenizer max length")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size")
    parser.add_argument("--max-wait-ms", type=int, default=30, help="Max wait milliseconds")
    parser.add_argument("--max-queue-size", type=int, default=1024, help="Max queue size")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU inference")
    args = parser.parse_args()

    service = BatchScoringService(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        max_queue_size=args.max_queue_size,
        allow_cpu=args.allow_cpu,
    )
    app = create_app(service)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
