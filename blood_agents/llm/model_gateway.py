"""只重试模型理解，不重放业务工具；完整提示词仅在本地 full 日志模式输出。"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass, field
from json import JSONDecodeError
from queue import Empty, Queue
from threading import Event, Lock, Thread

from pydantic import ValidationError

from blood_agents.log.trace_logging import trace_event


class ModelRefusalError(Exception):
    """模型显式拒答；不记录供应商原文。"""


def _safe_error_text(value: object) -> str:
    """本地 full 日志保留排障信息，但剔除密钥、鉴权字段和 URL 中的凭证。"""
    message = str(value).replace("\n", " ").replace("\r", " ")
    for variable in ("OPENAI_API_KEY", "BACKUP_OPENAI_API_KEY"):
        secret = os.getenv(variable, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)((?:api[_-]?key|access[_-]?token|authorization|password|secret)"
        r"\s*[=:]\s*['\"]?)[^\s&;,'\"}]+",
        r"\1[REDACTED]", message,
    )
    return message[:1500]


def _error_diagnostics(exc: Exception, *, completed: bool, wait_seconds: float) -> dict:
    """只供 full 追踪日志使用，不进入前端 model_info 或 summary 日志。"""
    if not completed:
        return {
            "timeout_source": "gateway_wait",
            "error_message": f"网关等待 {wait_seconds:.2f} 秒后模型请求仍未完成；底层请求可能仍在后台运行",
        }
    chain = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 4:
        seen.add(id(current))
        chain.append({"class": type(current).__name__, "message": _safe_error_text(current)})
        current = current.__cause__ or current.__context__
    details = {"error_message": chain[0]["message"], "cause_chain": chain}
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        details["http_status"] = status
    request_id = getattr(exc, "request_id", None)
    if request_id:
        details["request_id"] = _safe_error_text(request_id)
    return details


def failure_kind(exc: Exception) -> tuple[str, bool]:
    """第二项表示是否可以切备用；认证/参数/契约错误不盲目尝试其他模型。"""
    from openai import APIConnectionError, APITimeoutError

    if isinstance(exc, (TimeoutError, APITimeoutError)):
        return "timeout", True
    if isinstance(exc, (ConnectionError, APIConnectionError)):
        return "connection", True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate_limit", True
    if isinstance(status, int) and 500 <= status <= 599:
        return "server_error", True
    if status in (401, 403):
        return "authentication", False
    if status is not None:
        return "request_error", False
    if isinstance(exc, ModelRefusalError):
        return "model_refusal", False
    if isinstance(exc, (ValidationError, JSONDecodeError, TypeError)):
        return "invalid_model_output", False
    return "model_error", False


@dataclass
class ModelProvider:
    name: str
    model: object
    timeout: float = 10
    max_concurrency: int = 1
    failures: int = 0
    open_until: float = 0
    active: int = 0
    probe_active: bool = False
    lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self):
        """在数据类初始化后校验并整理运行配置。"""
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Model timeout must be positive and finite")
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int) or not 1 <= self.max_concurrency <= 64:
            raise ValueError("Model max_concurrency must be an integer from 1 to 64")


class GatewayFailure(Exception):
    def __init__(self, info):
        """初始化 GatewayFailure 的依赖与运行状态。"""
        super().__init__("Model services unavailable")
        self.info = info


class ModelGateway:
    """有界主备切换，进程内熔断；各服务有独立在途请求上限。

    同步 SDK 不支持强制取消：超时后不等待后台线程，但仍占用该服务的
    并发名额直至请求退出，防止悬挂请求不断堆积。生产客户端另设 HTTP 超时。
    """

    def __init__(self, providers, total_timeout=15, failure_threshold=2, cooldown=30):
        """初始化 ModelGateway 的依赖与运行状态。"""
        if not providers:
            raise ValueError("At least one model provider required")
        if any(not math.isfinite(x) or x <= 0 for x in (total_timeout, cooldown)):
            raise ValueError("Timeout and cooldown must be positive and finite")
        if failure_threshold < 1:
            raise ValueError("Failure threshold must be positive")
        self.providers = providers
        self.total_timeout = total_timeout
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown

    def call(self, messages, validate: Callable, stage: str = "unknown"):
        """调用指定业务工具并保留查询失败与写入异常的区别。"""
        started = time.monotonic()
        deadline = started + self.total_timeout
        attempts = []
        info = {"mode": "llm", "provider": None, "attempts": attempts, "error_type": None}
        for provider in self.providers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                info["error_type"] = "budget_exhausted"
                break
            # 原子占位同时限制正常并发和半开恢复探测；恢复时仅允许一个试探请求。
            with provider.lock:
                now = time.monotonic()
                recovering = provider.open_until > 0 and provider.failures >= self.failure_threshold
                skipped = (
                    "circuit_open" if provider.open_until > now else
                    "busy" if provider.active >= provider.max_concurrency or
                    (recovering and provider.active > 0) else None
                )
                if skipped:
                    attempts.append({"provider": provider.name, "status": skipped, "duration_ms": 0})
                    trace_event("model.attempt", summary={
                        "stage": stage, "provider": provider.name, "status": skipped,
                    })
                    continue
                provider.active += 1
                if recovering:
                    provider.probe_active = True
            done = Event()
            outcome = {}
            trace_event("model.attempt", summary={
                "stage": stage, "provider": provider.name, "status": "started",
                "timeout_seconds": provider.timeout,
            })

            # 默认参数冻结本次对象，超时线程不会误写下一次尝试的结果。
            def run(p=provider, result=outcome, finished=done):
                """在线程中执行供应商调用并记录最终状态。"""
                try:
                    reply = p.model.invoke(messages)
                    trace_event("model.response", summary={
                        "stage": stage, "provider": p.name,
                    }, full={"model_output": reply.content})
                    if getattr(reply, "additional_kwargs", {}).get("refusal"):
                        raise ModelRefusalError()
                    if not isinstance(reply.content, str) or not reply.content.strip():
                        raise TypeError("Model returned non-text or empty content")
                    result["value"] = validate(reply.content)
                except Exception as error:  # noqa: BLE001 - boundary to external model
                    result["error"] = error
                finally:
                    # 由等待方处理熔断之后再释放占位，避免恢复探测并发竞态。
                    finished.set()

            before = time.monotonic()
            # 网关使用后台线程实现超时；复制当前 ContextVar 才能给原始输出带上会话/轮次 ID。
            Thread(target=copy_context().run, args=(run,), daemon=True,
                   name="model-" + provider.name).start()
            wait_seconds = min(provider.timeout, max(0, deadline - time.monotonic()))
            completed = done.wait(wait_seconds)
            exc = outcome.get("error") if completed else TimeoutError()
            kind, retryable = failure_kind(exc) if exc else (None, False)
            with provider.lock:
                if exc and retryable:
                    provider.failures += 1
                    if provider.failures >= self.failure_threshold:
                        provider.open_until = time.monotonic() + self.cooldown
                elif not exc:
                    provider.failures = 0
                    provider.open_until = 0
                if completed:
                    provider.active -= 1
                    if recovering:
                        provider.probe_active = False
            if not completed:
                # 仅一个回收线程/服务；不等待超时 HTTP 返回，不产生无限在途请求。
                def release(p=provider, finished=done, result=outcome,
                            waited=wait_seconds, was_probe=recovering):
                    """释放请求占用的供应商在途计数。"""
                    finished.wait()
                    late_error = result.get("error")
                    trace_event("model.attempt.late_result", summary={
                        "stage": stage, "provider": p.name,
                        "status": "failed" if late_error else "success_ignored",
                        "error_type": failure_kind(late_error)[0] if late_error else None,
                        "exception_class": type(late_error).__name__ if late_error else None,
                    }, full=_error_diagnostics(late_error, completed=True,
                                               wait_seconds=waited) if late_error else None)
                    with p.lock:
                        p.active -= 1
                        if was_probe:
                            p.probe_active = False
                Thread(target=copy_context().run, args=(release,), daemon=True).start()
            attempts.append({
                "provider": provider.name, "status": "failed" if exc else "success",
                "error_type": kind,
                # 只公开异常类型名，便于区分解析失败与客户端错误；不输出异常原文。
                "exception_class": type(exc).__name__ if exc else None,
                "duration_ms": round((time.monotonic() - before) * 1000, 2),
            })
            trace_event("model.attempt.finished", summary={"stage": stage, **attempts[-1]},
                        full=_error_diagnostics(exc, completed=completed, wait_seconds=wait_seconds)
                        if exc else None)
            if not exc:
                info.update(provider=provider.name, duration_ms=round((time.monotonic() - started) * 1000, 2))
                return outcome["value"], info
            info["error_type"] = kind
            if not retryable:
                break
        info.update(mode="rules_fallback", duration_ms=round((time.monotonic() - started) * 1000, 2))
        info["error_type"] = info["error_type"] or "all_providers_unavailable"
        raise GatewayFailure(info)

    def stream(self, messages, validate: Callable, on_chunk: Callable[[str], None],
               stage: str = "unknown"):
        """逐块生成回复；首块发出后不再切备用，避免把两家模型的草稿拼在一起。"""
        started = time.monotonic()
        deadline = started + self.total_timeout
        attempts = []
        info = {"mode": "llm", "provider": None, "attempts": attempts, "error_type": None}
        for provider in self.providers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                info["error_type"] = "budget_exhausted"
                break
            with provider.lock:
                now = time.monotonic()
                recovering = provider.open_until > 0 and provider.failures >= self.failure_threshold
                skipped = (
                    "circuit_open" if provider.open_until > now else
                    "busy" if provider.active >= provider.max_concurrency or
                    (recovering and provider.active > 0) else None
                )
                if skipped:
                    attempts.append({"provider": provider.name, "status": skipped, "duration_ms": 0})
                    continue
                provider.active += 1
                if recovering:
                    provider.probe_active = True

            events: Queue = Queue()
            finished, cancelled = Event(), Event()
            assembled: list[str] = []
            chunk_count = 0
            output_chars = 0
            first_chunk_ms = None

            def produce(p=provider, out=events, done=finished, stop=cancelled):
                """在后台发起一次模型请求并保存候选结果。"""
                try:
                    for chunk in p.model.stream(messages):
                        if stop.is_set():
                            continue
                        content = getattr(chunk, "content", "")
                        if isinstance(content, str) and content:
                            out.put(("chunk", content))
                    out.put(("done", None))
                except Exception as caught_error:  # noqa: BLE001 - provider boundary
                    out.put(("error", caught_error))
                finally:
                    done.set()

            trace_event("model.attempt", summary={
                "stage": stage, "provider": provider.name, "status": "started_stream",
                "timeout_seconds": provider.timeout,
            })
            before = time.monotonic()
            Thread(target=copy_context().run, args=(produce,), daemon=True,
                   name="model-stream-" + provider.name).start()
            wait_budget = min(provider.timeout, max(0, deadline - time.monotonic()))
            stream_deadline = time.monotonic() + wait_budget
            exc = None
            completed = False
            emitted = False
            while True:
                wait_left = stream_deadline - time.monotonic()
                if wait_left <= 0:
                    exc = TimeoutError()
                    break
                try:
                    kind, value = events.get(timeout=wait_left)
                except Empty:
                    exc = TimeoutError()
                    break
                if kind == "chunk":
                    assembled.append(value)
                    emitted = True
                    chunk_count += 1
                    output_chars += len(value)
                    if first_chunk_ms is None:
                        first_chunk_ms = round((time.monotonic() - before) * 1000, 2)
                        trace_event("model.stream.first_chunk", summary={
                            "stage": stage, "provider": provider.name,
                            "first_chunk_ms": first_chunk_ms,
                        })
                    try:
                        on_chunk(value)
                    except Exception as callback_error:  # noqa: BLE001 - UI 断开不泄露/重试文本
                        exc = callback_error
                        break
                    continue
                if kind == "error":
                    exc = value
                    completed = finished.is_set()
                else:
                    completed = True
                break

            if not exc:
                raw = "".join(assembled)
                try:
                    if not raw.strip():
                        raise TypeError("Model returned non-text or empty content")
                    result = validate(raw)
                    trace_event("model.response", summary={
                        "stage": stage, "provider": provider.name, "stream": True,
                        "chunk_count": chunk_count, "output_chars": output_chars,
                        "first_chunk_ms": first_chunk_ms,
                    }, full={"model_output": raw})
                    info.update(provider=provider.name,
                                duration_ms=round((time.monotonic() - started) * 1000, 2),
                                first_chunk_ms=first_chunk_ms,
                                chunk_count=chunk_count, output_chars=output_chars)
                except Exception as error:  # noqa: BLE001 - output contract boundary
                    exc = error

            kind, retryable = failure_kind(exc) if exc else (None, False)
            with provider.lock:
                if exc and retryable:
                    provider.failures += 1
                    if provider.failures >= self.failure_threshold:
                        provider.open_until = time.monotonic() + self.cooldown
                elif not exc:
                    provider.failures = 0
                    provider.open_until = 0
                if completed:
                    provider.active -= 1
                    if recovering:
                        provider.probe_active = False

            if not completed:
                cancelled.set()
                def release(p=provider, done=finished, was_probe=recovering):
                    """释放请求占用的供应商在途计数。"""
                    done.wait()
                    with p.lock:
                        p.active -= 1
                        if was_probe:
                            p.probe_active = False
                Thread(target=copy_context().run, args=(release,), daemon=True,
                       name="model-stream-release-" + provider.name).start()

            attempts.append({
                "provider": provider.name, "status": "failed" if exc else "success",
                "error_type": kind, "exception_class": type(exc).__name__ if exc else None,
                "duration_ms": round((time.monotonic() - before) * 1000, 2),
                "first_chunk_ms": first_chunk_ms, "chunk_count": chunk_count,
                "output_chars": output_chars,
            })
            trace_event("model.attempt.finished", summary={"stage": stage, **attempts[-1]},
                        full=_error_diagnostics(exc, completed=completed,
                                                wait_seconds=wait_budget) if exc else None)
            if not exc:
                return result, info
            info["error_type"] = kind
            # 只允许“尚无任何可见文本”的失败切备用；否则无法撤回用户已看到的草稿。
            if emitted or not retryable:
                break

        info.update(mode="unavailable", duration_ms=round((time.monotonic() - started) * 1000, 2))
        info["error_type"] = info["error_type"] or "all_providers_unavailable"
        raise GatewayFailure(info)
