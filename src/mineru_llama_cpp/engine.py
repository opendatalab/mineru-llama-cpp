"""The Engine class. See design spec §5.4."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import TracebackType
from typing import AsyncIterator, Callable, Iterator, TypeVar

from ._mineru_llama_cpp import _EngineCore
from .exceptions import EngineClosedError, EngineUnavailableError
from .sampling import SamplingParams
from .types import GenerateChunk, GenerateResult, GenerationTimings, Messages
from .verbosity import LOG_LEVEL_WARN

# "any Unicode codepoint, repeated" -- llama.cpp's grammar-constrained sampler
# decodes each candidate token's bytes and rejects invalid UTF-8 sequences
# (see llama-grammar.cpp's decode_utf8()/llama_grammar_match_partial_char()).
# Valid characters may span tokens, so a forced length stop still needs
# boundary-safe decoding in the binding layer. Occasionally (observed on
# a Q8_0-quantized model, dense/high-resolution pages) the model samples a
# token that produces a broken multi-byte sequence, which renders as U+FFFD
# and breaks the downstream peg-native chat-format parser
# ("The model produced output that does not match the expected ... format").
# This wildcard grammar doesn't constrain the *structure* of the output
# (MinerU's <|box_start|> etc. markers and content are still whatever the
# model produces) -- it only removes the invalid-byte failure mode from the
# sampling candidate set.
#
# Applied in _build_body (below) as a *default*, the same way SamplingParams
# .stop is defaulted: only set when the caller hasn't already put a "grammar"
# key in the body. SamplingParams has no grammar field today (see its module
# docstring), so in practice this is always the effective value -- but if a
# grammar field is added to SamplingParams later, that value should win over
# this default rather than being silently overridden by it.
_VALID_UNICODE_GRAMMAR = "root ::= .*"


def _timings_from_dict(d: dict | None) -> GenerationTimings | None:
    if d is None:
        return None
    return GenerationTimings(**d)


_Result = TypeVar("_Result")


async def _wait_until_complete(
    work: asyncio.Future[_Result], cancelled: threading.Event | None = None
) -> _Result:
    """取消后排空已提交线程，明确死亡优先上报，其他情形保留调用方取消。"""
    try:
        await asyncio.wait([work])
        return work.result()
    except asyncio.CancelledError:
        if cancelled is not None:
            cancelled.set()
        while not work.done():
            try:
                await asyncio.wait([work])
            except asyncio.CancelledError:
                continue
        if not work.cancelled():
            error = work.exception()
            if isinstance(error, EngineUnavailableError):
                raise error
        raise


async def _run_in_thread(function: Callable[[], _Result]) -> _Result:
    """未开始的工作取消后跳过执行；已开始的工作等待结束再交还控制权。"""
    cancelled = threading.Event()

    def run() -> _Result | None:
        """在线程开始时检查取消，避免排队任务在取消后启动迟到推理。"""
        if cancelled.is_set():
            return None
        return function()

    return await _wait_until_complete(
        asyncio.get_running_loop().run_in_executor(None, run), cancelled
    )


class Engine:
    def __init__(
        self,
        model: str | Path,
        mmproj: str | Path,
        *,
        n_ctx_seq: int = 0,
        n_gpu_layers: int = 99,
        n_parallel: int = 4,
        verbosity: int = LOG_LEVEL_WARN,
        n_threads: int = -1,
    ) -> None:
        """Loads the model and starts the background decode loop thread.

        model/mmproj must be local file paths (no HuggingFace repo id
        auto-download — see design spec §1 "非目标").

        n_ctx_seq: the context length for a *single* slot/sequence (this maps
            onto llama.cpp's internal n_ctx_seq, "context for a single
            sequence" -- not the total n_ctx). The total context is derived
            internally as n_ctx_seq * n_parallel, and the KV cache is
            hard-partitioned (kv_unified=false) so each slot gets its own
            private n_ctx_seq cells -- no slot can be starved by its peers.
            Default 0 means "the model's training context length"
            (n_ctx_train), so most callers only ever need to tune n_parallel.
            Values are rounded up to a multiple of 256 to match llama.cpp's
            own internal padding, keeping the effective per-slot context
            exactly equal to what's passed here.
        n_parallel: number of concurrent slots, same as llama.cpp's
            -np/--parallel. Default 4 matches what llama-server itself
            resolves to when --parallel is left at its own "auto" default
            (tools/server/server.cpp). Total KV = n_ctx_seq * n_parallel.
        verbosity: threshold for llama.cpp's internal logging, using the
            same LOG_LEVEL_* constants as llama.cpp's own -lv/--verbosity
            CLI flag (see mineru_llama_cpp.verbosity). Default is
            LOG_LEVEL_WARN (llama.cpp's own default is LOG_LEVEL_INFO,
            which is noisier than most embedders want -- pass
            LOG_LEVEL_INFO explicitly to match llama.cpp's own default).
        n_threads: CPU threads for generation, same as llama.cpp's
            -t/--threads. Default -1 auto-detects the usable core count.
        """
        # Set before constructing _core: if _EngineCore(...) raises, __del__
        # on the partially-constructed Engine still finds these attributes
        # (otherwise close() would hit AttributeError, silently swallowed by
        # __del__'s try/except, masking the real construction failure).
        self._closed = False
        self._close_lock = threading.Lock()
        # Validate at the boundary: a negative n_ctx_seq or a non-positive
        # n_parallel would otherwise produce a nonsensical total n_ctx (or a
        # cryptic assert deep inside llama.cpp) rather than a clear error here.
        # n_ctx_seq == 0 is valid and means "use the model's training context".
        if n_ctx_seq < 0:
            raise ValueError(
                f"n_ctx_seq must be >= 0 (0 = model training context), got {n_ctx_seq}"
            )
        if n_parallel < 1:
            raise ValueError(f"n_parallel must be >= 1, got {n_parallel}")
        # Round a nonzero n_ctx_seq up to a multiple of 256 so it survives
        # llama.cpp's own GGML_PAD(x, 256) untouched -- otherwise the C++ side
        # would pad n_ctx/n_parallel internally and the effective per-slot
        # context would silently differ from what was passed. 0 is left as-is
        # (the C++ side resolves it to n_ctx_train, itself a multiple of 256).
        if n_ctx_seq > 0 and n_ctx_seq % 256 != 0:
            n_ctx_seq = ((n_ctx_seq + 255) // 256) * 256
        self._core = _EngineCore(
            str(model),
            str(mmproj),
            n_ctx_seq,
            n_gpu_layers,
            n_parallel,
            verbosity,
            n_threads,
        )
        # The engine is constructed with special=true (see engine_core.cpp),
        # which is needed to keep MinerU's structured tokens (<|box_start|>
        # etc.) intact -- but that also means the chat template's own EOS
        # token gets rendered as literal text into `content` instead of
        # being silently dropped, e.g. "...<|im_end|>" trailing the output.
        # Auto-filling SamplingParams.stop with it (when the caller hasn't
        # set their own) strips it the same way llama-server would for any
        # explicit stop string -- see server-context.cpp's
        # find_stopping_strings()/erase() in process_token().
        self._eos_token_str: str = self._core.eos_token_str

    def _build_body(
        self, messages: Messages, sampling_params: SamplingParams | None, stream: bool
    ) -> str:
        body: dict = {
            "model": "mineru-llama-cpp",
            "messages": messages,
            "stream": stream,
        }
        if sampling_params is not None:
            body.update(sampling_params.to_json_fields())
        # Default, not an override -- see _VALID_UNICODE_GRAMMAR's
        # module-level comment. A future caller-supplied "grammar" would win.
        if "grammar" not in body:
            body["grammar"] = _VALID_UNICODE_GRAMMAR
        if "stop" not in body and self._eos_token_str:
            body["stop"] = [self._eos_token_str]
        return json.dumps(body)

    # --- non-streaming ---

    def generate(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        """Blocking call. Raises InvalidRequestError/ContextExceededError/
        EngineError on failure (see exceptions.py)."""
        core = self._require_core()
        body = self._build_body(messages, sampling_params, stream=False)
        d = core.generate(body)
        return GenerateResult(
            content=d["content"],
            finish_reason=d["finish_reason"],
            tokens_evaluated=d["tokens_evaluated"],
            tokens_predicted=d["tokens_predicted"],
            timings=_timings_from_dict(d["timings"]),
        )

    async def agenerate(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        """Async version of generate(), bridged via run_in_executor — does
        not block the event loop."""
        return await _run_in_thread(lambda: self.generate(messages, sampling_params))

    # --- streaming ---

    def stream(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[GenerateChunk]:
        """Synchronous generator, yields one GenerateChunk per token. The
        final chunk has finish_reason set (all others have it as None) —
        check `chunk.finish_reason is not None` to detect the end."""
        core = self._require_core()
        body = self._build_body(messages, sampling_params, stream=True)
        core_iter = core.generate_stream(body)
        for c in core_iter:
            yield GenerateChunk(
                delta=c["delta"],
                finish_reason=c["finish_reason"],
                tokens_evaluated=c["tokens_evaluated"],
                tokens_predicted=c["tokens_predicted"],
                timings=_timings_from_dict(c["timings"]),
            )

    async def astream(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        """Async version of stream(): a real generator-to-generator bridge.

        run_in_executor() alone can't drive a synchronous *generator* (it's
        built for "call once, get one result"), so this runs stream() to
        completion on a background thread and relays each chunk back to the
        event loop through an asyncio.Queue via call_soon_threadsafe — the
        standard sync-iterator-to-async-iterator bridge pattern.

        Note: breaking out of the consuming loop early (or otherwise
        abandoning this generator before it's exhausted) does not cancel
        the underlying generation — the background thread runs to
        completion regardless, occupying a slot until the model finishes.
        Explicit aclose() or cancellation waits for that work to drain.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        abandoned = threading.Event()
        work: asyncio.Future[None] = loop.create_future()

        def _run() -> None:
            """在工作线程生成，消费者退出后仅排空而不继续缓存文本。"""
            fatal: EngineUnavailableError | None = None
            try:
                for chunk in self.stream(messages, sampling_params):
                    if not abandoned.is_set():
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:  # noqa: BLE001 - relayed to the consumer, not swallowed
                loop.call_soon_threadsafe(queue.put_nowait, exc)
                if isinstance(exc, EngineUnavailableError):
                    fatal = exc
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
                if fatal is not None:
                    loop.call_soon_threadsafe(work.set_exception, fatal)
                else:
                    loop.call_soon_threadsafe(work.set_result, None)

        # 保留独立流线程，避免同步流占满共享 executor 后阻塞其他异步调用。
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            abandoned.set()
            await _wait_until_complete(work)

    # --- lifecycle ---

    def _require_core(self) -> _EngineCore:
        """在读取或生成前拒绝已关闭实例，并持有局部引用直至本次调用排空。"""
        with self._close_lock:
            if self._closed:
                raise EngineClosedError(
                    "engine is closed", error_type="engine_closed", code="engine_closed"
                )
            return self._core

    def check_health(self) -> None:
        """查询原生循环终止状态；活着但卡住的循环不伪称已死亡，不发送推理任务。"""
        self._require_core().check_health()

    def close(self) -> None:
        """Explicit shutdown: terminate + join the background loop thread,
        free the llama backend. Idempotent and thread-safe (safe to call
        more than once, including concurrently — aclose() offloads to a
        thread pool executor, making concurrent close() calls a realistic
        scenario, not just a hypothetical one).

        v1 assumes no in-flight generate()/stream() calls when close() is
        called — see design spec §5.4's note on close()/in-flight requests
        for why this isn't handled defensively yet."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            # drops the last reference -> EngineCore's C++ destructor runs now
            del self._core

    async def aclose(self) -> None:
        """Async version of close(), offloaded to a thread so it doesn't
        block the event loop while the loop thread joins."""
        loop = asyncio.get_event_loop()
        await _wait_until_complete(loop.run_in_executor(None, self.close))

    def __enter__(self) -> Engine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Engine:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def __del__(self) -> None:
        """Safety-net fallback only — normal code should call close()/
        aclose() explicitly (or use `with`/`async with`)."""
        try:
            self.close()
        except Exception:
            pass
