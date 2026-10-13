"""
Token 用量与成本统计（进程内全局累加器）

用途：
- 量化「精准决策」的成本指标（申报书/评测报告需要真实数据，不能估算）
- 每次 LLM 调用后调用 record(response) 即可自动累加

计费口径（DeepSeek 官方标准价，2026）：
- 输入 ¥2 / 百万 token（缓存未命中）
- 输出 ¥8 / 百万 token
实际费用因 prompt 缓存命中而更低，此处按标准价上限估算，口径保守。
"""
import threading
from contextlib import contextmanager

_lock = threading.Lock()
_state = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

COST_PER_M_INPUT = 2.0    # 元 / 百万 token
COST_PER_M_OUTPUT = 8.0   # 元 / 百万 token


def record(response) -> None:
    """记录一次 LLM 调用的 token 用量（传 OpenAI 兼容的 response 对象）"""
    usage = getattr(response, "usage", None)
    if not usage:
        return
    with _lock:
        _state["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _state["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        _state["calls"] += 1


def reset() -> None:
    with _lock:
        _state.update({"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})


def snapshot() -> dict:
    """返回当前累计用量与费用"""
    with _lock:
        s = dict(_state)
    cost = (s["prompt_tokens"] / 1e6) * COST_PER_M_INPUT + \
           (s["completion_tokens"] / 1e6) * COST_PER_M_OUTPUT
    return {
        "prompt_tokens": s["prompt_tokens"],
        "completion_tokens": s["completion_tokens"],
        "total_tokens": s["prompt_tokens"] + s["completion_tokens"],
        "llm_calls": s["calls"],
        "cost_cny": round(cost, 6),
    }


@contextmanager
def measure():
    """上下文管理器：自动 reset 并在退出时返回本次用量（通过 yield 的 dict 读取）"""
    reset()
    result = {}
    try:
        yield result
    finally:
        result.update(snapshot())
