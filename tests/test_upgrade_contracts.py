"""验证上游升级后 JSON、流式边界和统计字段的 Python 契约。"""

import asyncio
import json

import pytest

from mineru_llama_cpp import InvalidRequestError, SamplingParams


@pytest.mark.parametrize("body", ["{broken", "[]", "null", '{"messages":"wrong type"}'])
def test_invalid_json_request_does_not_poison_engine(engine, body):
    """无效 JSON 或错误字段类型仍报请求异常，随后正常请求可继续运行。"""
    with pytest.raises(InvalidRequestError):
        engine._core.generate(body)
    result = engine.generate([{"role": "user", "content": "hi"}], SamplingParams(n_predict=4))
    assert result.content


def test_chat_session_applies_template_grammar_to_generate_and_stream(engine):
    """验证会话产生的 JSON Schema 约束在整段和流式请求中均交给采样器。"""
    schema = {"type": "object", "properties": {"value": {"const": "sync"}},
              "required": ["value"], "additionalProperties": False}
    body = {
        "messages": [{"role": "user", "content": "Name a fruit."}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}},
        "temperature": 0.0,
        "max_tokens": 32,
        "stop": [engine._core.eos_token_str],
    }
    result = engine._core.generate(json.dumps(body))
    assert json.loads(result["content"]) == {"value": "sync"}
    chunks = engine._core.generate_stream(json.dumps({**body, "stream": True}))
    assert json.loads("".join(chunk["delta"] for chunk in chunks)) == {"value": "sync"}


def test_serialized_timings_and_stream_boundaries(engine):
    """新上游序列化结果保留计数、统计和空起始块，流式文本与整段文本一致。"""
    messages = [{"role": "user", "content": "List three fruits."}]
    sampling = SamplingParams(n_predict=8, temperature=0.0, top_k=1, seed=42)
    result = engine.generate(messages, sampling)
    chunks = list(engine.stream(messages, sampling))
    assert chunks[0].delta == ""
    assert chunks[0].finish_reason is None
    assert "".join(chunk.delta for chunk in chunks) == result.content
    final = chunks[-1]
    assert final.finish_reason == result.finish_reason
    assert final.tokens_evaluated == result.tokens_evaluated
    assert final.tokens_predicted == result.tokens_predicted
    for value in (result, final):
        assert value.timings is not None
        assert value.timings.prompt_n == value.tokens_evaluated
        assert value.timings.predicted_n == value.tokens_predicted
        assert value.timings.prompt_ms >= 0
        assert value.timings.predicted_ms >= 0


async def test_cancelled_waiter_does_not_poison_slots(engine):
    """取消异步等待保持现有后台完成语义，其他请求与后续流式仍可运行。"""
    sampling = SamplingParams(n_predict=32, temperature=0.0, top_k=1)
    messages = [{"role": "user", "content": "List three fruits."}]
    task = asyncio.create_task(engine.agenerate(messages, sampling))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await engine.agenerate(messages, sampling)
    assert result.content
    chunks = [chunk async for chunk in engine.astream(messages, sampling)]
    # 基线同样会保留潜在停止串前缀；此用例检查取消后的槽位和结束元数据。
    assert "".join(chunk.delta for chunk in chunks)
    assert chunks[-1].finish_reason in ("stop", "length")
    assert chunks[-1].tokens_predicted > 0
