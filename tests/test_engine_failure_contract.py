"""验证实际公开异常类、关闭拒绝及异步线程排空契约。"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mineru_llama_cpp import (
    Engine,
    EngineClosedError,
    EngineError,
    EngineUnavailableError,
    InvalidRequestError,
)


@pytest.fixture
def empty_engine(monkeypatch: pytest.MonkeyPatch) -> Engine:
    """只替换模型核心，保留生产 Engine 的公共边界和异步实现。"""
    import mineru_llama_cpp.engine as module

    monkeypatch.setattr(
        module, "_EngineCore", lambda *args: SimpleNamespace(eos_token_str="")
    )
    return Engine("unused", "unused")


def test_structured_error_hierarchy() -> None:
    """普通服务错误不代表实例死亡，明确关闭同时属于致命及服务错误。"""
    ordinary = EngineError(
        "busy", code=503, error_type="unavailable_error", details={"param": "slot"}
    )
    assert not isinstance(ordinary, EngineUnavailableError)
    assert ordinary.code == 503 and ordinary.details == {"param": "slot"}
    assert not issubclass(InvalidRequestError, EngineError)
    assert issubclass(EngineClosedError, EngineUnavailableError)


def test_closed_engine_rejects_every_entry(empty_engine: Engine) -> None:
    """显式关闭后健康、同步及生成器入口均返回精确错误，重复关闭安全。"""
    empty_engine.close()
    empty_engine.close()
    for call in [
        empty_engine.check_health,
        lambda: empty_engine.generate([]),
        lambda: next(empty_engine.stream([])),
    ]:
        with pytest.raises(EngineClosedError) as captured:
            call()
        assert captured.value.code == "engine_closed"


@pytest.mark.asyncio
async def test_closed_async_entries(empty_engine: Engine) -> None:
    """异步与同步使用同一关闭边界。"""
    empty_engine.close()
    with pytest.raises(EngineClosedError):
        await empty_engine.agenerate([])
    with pytest.raises(EngineClosedError):
        await anext(empty_engine.astream([]))


@pytest.mark.asyncio
@pytest.mark.parametrize("fatal", [False, True])
async def test_cancel_drains_generation(empty_engine: Engine, fatal: bool) -> None:
    """重复取消不越过线程排空，排空期间的确认死亡仍能上报批次收集器。"""
    started, release, ended = threading.Event(), threading.Event(), threading.Event()
    error = EngineUnavailableError("loop stopped", code="engine_dead")

    def generate(*args):
        """模拟已进入原生推理的线程，在测试许可后退出。"""
        started.set()
        try:
            assert release.wait(5)
            if fatal:
                raise error
            return None
        finally:
            ended.set()

    empty_engine.generate = generate
    task = asyncio.create_task(empty_engine.agenerate([]))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not ended.is_set()
    release.set()
    with pytest.raises(EngineUnavailableError if fatal else asyncio.CancelledError):
        await task
    assert ended.is_set()


def test_cancelled_queue_never_starts_inference(empty_engine: Engine) -> None:
    """独立循环中占满线程池，验证排队期间取消不会迟到调用引擎。"""
    started, release = threading.Event(), threading.Event()
    calls = []
    empty_engine.generate = lambda *args: calls.append(args)

    def occupy() -> None:
        """占用唯一线程，直到取消已生效。"""
        started.set()
        assert release.wait(5)

    async def scenario() -> None:
        """在独立默认线程池上控制提交与取消顺序。"""
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        busy = loop.run_in_executor(None, occupy)
        while not started.is_set():
            await asyncio.sleep(0.001)
        task = asyncio.create_task(empty_engine.agenerate([]))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await busy

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.asyncio
async def test_astream_close_drains_worker(empty_engine: Engine) -> None:
    """显式关闭异步生成器时等待同步流排空，不能提前释放模型。"""
    release, ended = threading.Event(), threading.Event()

    def stream(*args):
        """首片输出后阻塞，模拟仍占用模型的原生流。"""
        yield "first"
        assert release.wait(5)
        ended.set()

    empty_engine.stream = stream
    stream_iter = empty_engine.astream([])
    assert await anext(stream_iter) == "first"
    closing = asyncio.create_task(stream_iter.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    assert ended.is_set()


@pytest.mark.asyncio
async def test_astream_fatal_during_close_is_preserved(empty_engine: Engine) -> None:
    """流消费者停止后才到达的致命异常仍要交给宿主，不能留在无人读取的队列。"""
    release = threading.Event()
    original = EngineUnavailableError("loop failed while draining", code="engine_dead")

    def stream(*args):
        """在首片输出之后、消费者关闭期间模拟真实类型的终止故障。"""
        yield "first"
        assert release.wait(5)
        raise original

    empty_engine.stream = stream
    iterator = empty_engine.astream([])
    assert await anext(iterator) == "first"
    closing = asyncio.create_task(iterator.aclose())
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(EngineUnavailableError) as raised:
        await closing
    assert raised.value is original


@pytest.mark.asyncio
async def test_aclose_cancellation_waits_for_release(empty_engine: Engine) -> None:
    """取消异步关闭不能让调用方误判原生资源已经释放。"""
    started, release, ended = threading.Event(), threading.Event(), threading.Event()

    def close() -> None:
        """用可控释放过程模拟等待原生后台线程退出。"""
        started.set()
        assert release.wait(5)
        ended.set()

    empty_engine.close = close
    closing = asyncio.create_task(empty_engine.aclose())
    assert await asyncio.to_thread(started.wait, 5)
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert ended.is_set()
