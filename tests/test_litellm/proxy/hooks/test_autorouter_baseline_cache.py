import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, cast

import httpx
import pytest
import respx
from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.autorouter_baseline_cache import (
    AutoRouterBaselineCache,
    BaselineCacheContext,
    cancel_baseline_cache,
    finalize_baseline_cache,
    invalidate_baseline_cache,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimator,
    BaselineReservation,
    _HistoryStore,  # pyright: ignore[reportPrivateUsage]  # inject faults into the existing storage dependency
    _Snapshot,  # pyright: ignore[reportPrivateUsage]  # retain the storage operation's typed contract
    _State,  # pyright: ignore[reportPrivateUsage]  # retain the storage operation's typed contract
    _StoreFailure,  # pyright: ignore[reportPrivateUsage]  # distinguish returned storage faults from raised exceptions
    unknown_estimate,
)
from litellm.proxy.utils import InternalUsageCache, ProxyLogging
from litellm.router import Router
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.router import RetryPolicy
from litellm.types.utils import CallTypes, ModelResponse, StandardLoggingRoutingDecision, Usage

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OBJECTS: Final = TypeAdapter(dict[str, object])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_MESSAGES_JSON: Final = """[{"role":"user","content":[
    {"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}},
    {"type":"text","text":"question"}]}]"""
_THINKING_MESSAGES: Final = """[
    {"role":"user","content":[{"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
    {"role":"assistant","content":[{"type":"thinking","thinking":"reasoning","signature":"invalid"},{"type":"text","text":"answer"}]},
    {"role":"user","content":"question"}]"""
_MODELS: Final = _MESSAGES.validate_json("""[
    {"model_name":"test-router","litellm_params":{"model":"auto_router/complexity_router",
      "complexity_router_config":{"tiers":{"SIMPLE":"sonnet","MEDIUM":"sonnet","COMPLEX":"sonnet",
      "REASONING":"opus"},"session_affinity":false}}},
    {"model_name":"sonnet","litellm_params":{"model":"anthropic/claude-sonnet-5","api_key":"test-selected"},
      "model_info":{"id":"selected"}},
    {"model_name":"opus","litellm_params":{"model":"anthropic/claude-opus-5","api_key":"test-selected"},
      "model_info":{"id":"baseline"}}]""")


def _message(completed: bool) -> Mapping[str, JsonValue]:
    return _JSON_OBJECT.validate_json(f"""{{
        "id":"msg_baseline_test","type":"message","role":"assistant","model":"claude-sonnet-5",
        "content":{'[{"type":"text","text":"OK"}]' if completed else "[]"},
        "stop_reason":{'"end_turn"' if completed else "null"},"stop_sequence":null,
        "usage":{{"input_tokens":1000,"output_tokens":{10 if completed else 0},
          "cache_creation_input_tokens":5000,"cache_read_input_tokens":0,
          "cache_creation":{{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":5000}}}}}}""")


_COMPLETED: Final = _message(True)
_START_MESSAGE: Final = _message(False)
_EVENTS: Final = (
    _JSON_OBJECT.validate_python(MappingProxyType({"type": "message_start", "message": _START_MESSAGE})),
    _JSON_OBJECT.validate_json('{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'),
    _JSON_OBJECT.validate_json('{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK"}}'),
    _JSON_OBJECT.validate_json('{"type":"content_block_stop","index":0}'),
    _JSON_OBJECT.validate_json(
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}'
    ),
    _JSON_OBJECT.validate_json('{"type":"message_stop"}'),
)
pytestmark: Final = pytest.mark.asyncio


class _Capture(CustomLogger):
    def __init__(self) -> None:
        self.payloads: asyncio.Queue[Mapping[str, object]] = asyncio.Queue()
        self.failures: asyncio.Queue[tuple[Mapping[str, object], Exception]] = asyncio.Queue()
        self.terminal_errors: asyncio.Queue[Exception] = asyncio.Queue()
        self.attempts: tuple[Logging, ...] = ()
        self.replacement_error = HTTPException(status_code=429, detail="transformed terminal provider error")

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("litellm_call_id") in ("first", "following", "matching"):
            self.payloads.put_nowait(_OBJECTS.validate_python(kwargs.get("standard_logging_object")))

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("litellm_call_id") == "first":
            exception: Final = kwargs.get("exception")
            assert isinstance(exception, Exception)
            self.failures.put_nowait((_OBJECTS.validate_python(kwargs.get("standard_logging_object")), exception))

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if isinstance(logging_obj, Logging) and call_type == CallTypes.anthropic_messages:
            self.attempts = (*self.attempts, logging_obj)

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> HTTPException:
        assert "litellm_logging_obj" not in request_data
        self.terminal_errors.put_nowait(original_exception)
        return self.replacement_error

    async def payload(self) -> Mapping[str, object]:
        return await asyncio.wait_for(self.payloads.get(), timeout=20)


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.failures_remaining = 0
        self.failure_at: int | None = None
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls == self.failure_at or self.failures_remaining:
            self.failures_remaining = max(0, self.failures_remaining - 1)
            raise ValueError("injected estimator clock failure")
        return self.now


class _FailingHistoryStore(_HistoryStore):
    def __init__(self) -> None:
        super().__init__(DualCache())
        self.fail_next_exchange: Literal["before", "after"] | None = None

    async def exchange(self, scope: str, before: _Snapshot, after: _State, now: float) -> bool | _StoreFailure:
        fault: Final = self.fail_next_exchange
        self.fail_next_exchange = None
        if fault == "before":
            return _StoreFailure()
        applied: Final = await super().exchange(scope, before, after, now)
        return _StoreFailure() if fault == "after" else applied


class _DelayedCancelEstimator(BaselineCacheEstimator):
    def __init__(self, clock: _Clock, gate: "_CountingGate") -> None:
        super().__init__(DualCache(), token_counter=gate.count, clock=clock)
        self.gate = gate

    async def cancel(self, reservation: BaselineReservation) -> None:
        await self.gate.wait()
        await super().cancel(reservation)


async def _count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
    assert model == "claude-opus-5"
    return 6000 if "question" in json.dumps(_JSON_OBJECT.validate_python(body)) else 5000


class _CountingGate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.started.set()
        await self.release.wait()

    async def count(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        await self.wait()
        return await _count(model, api_key, body)


@dataclass(frozen=True, slots=True)
class _Rig:
    router: Router
    estimator: BaselineCacheEstimator
    hook: AutoRouterBaselineCache
    capture: _Capture


def _rig(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock | None = None, retries: int = 0, lookup_fails: bool = False
) -> _Rig:
    router: Final = Router(
        model_list=_MODELS,
        num_retries=retries,
        retry_policy=RetryPolicy(RateLimitErrorRetries=retries),
        disable_cooldowns=True,
    )

    def get_router() -> Router:
        if lookup_fails:
            raise ValueError("injected baseline deployment lookup failure")
        return router

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count, clock=clock or time.time)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    capture: Final = _Capture()
    for name in ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "callbacks", [hook, capture])  # mutable-ok: LiteLLM mutates callback registries
    for name in ("success_callback", "failure_callback"):
        monkeypatch.setattr(litellm, name, [])  # mutable-ok: LiteLLM mutates callback registries
    for name in ("_async_success_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, [capture])  # mutable-ok: LiteLLM mutates callback registries
    return _Rig(router, estimator, hook, capture)


def _logging(request_id: str = "first", stream: bool = False) -> Logging:
    return Logging(  # pyright: ignore[reportUnknownMemberType]  # production constructor has legacy argument types
        model="anthropic/claude-sonnet-5",
        messages=_MESSAGES.validate_json(_MESSAGES_JSON),
        stream=stream,
        call_type=CallTypes.anthropic_messages.value,
        start_time=datetime.now(),  # noqa: DTZ005  # native Logging requires naive timestamps
        litellm_call_id=request_id,
        function_id=request_id,
        kwargs=_OBJECTS.validate_json('{"litellm_session_id":"baseline-session"}'),
    )


def _metadata(trusted: bool = True) -> Mapping[str, object]:
    kwargs: Final = _OBJECTS.validate_json('{"litellm_metadata":{"user_api_key_hash":"test-caller-hash"}}')
    Router._record_routing_decision(  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # production trusted stamp owner
        kwargs,
        StandardLoggingRoutingDecision(
            router_model_name="test-router",
            router_type="complexity",
            routed_model="sonnet",
            cause="heuristic_scorer",
            conversation_continuing=True,
            savings_baseline_model="anthropic/claude-opus-5",
            savings_baseline_deployment_id="baseline",
        ),
    )
    metadata: Final = _OBJECTS.validate_python(kwargs["litellm_metadata"])
    if trusted:
        return metadata
    return dict(  # mutable-ok: legacy metadata lookup requires a concrete dictionary
        metadata,
        _autorouter_baseline_route=_JSON_OBJECT.validate_json(
            '{"router_name":"test-router","baseline_model":"anthropic/claude-opus-5","baseline_deployment_id":"baseline"}'
        ),
    )


class _CallContext(TypedDict):
    litellm_logging_obj: NotRequired[ReadOnly[Logging]]
    litellm_call_id: ReadOnly[str]
    litellm_metadata: ReadOnly[Mapping[str, object]]
    litellm_session_id: ReadOnly[str]


def _kwargs(logging_obj: Logging, trusted: bool = True, *, explicit_logging: bool = True) -> _CallContext:
    context: Final[_CallContext] = {
        "litellm_call_id": logging_obj.litellm_call_id,
        "litellm_metadata": _metadata(trusted),
        "litellm_session_id": "baseline-session",
    }
    if not explicit_logging:
        return context
    supplied: Final[_CallContext] = {**context, "litellm_logging_obj": logging_obj}
    return supplied


def _stream(logging_obj: Logging) -> bool:
    return logging_obj.stream is True  # pyright: ignore[reportUnknownMemberType]  # normalize the legacy Logging flag


def _sse(completed: bool = True) -> tuple[bytes, ...]:
    return tuple(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        for event in (_EVENTS if completed else _EVENTS[:-1])
    )


def _upstream(request: httpx.Request) -> httpx.Response:
    if _JSON_OBJECT.validate_json(request.content).get("stream") is True:
        return httpx.Response(
            200,
            content=b"".join(_sse()),
            headers=MappingProxyType({"content-type": "text/event-stream"}),
            request=request,
        )
    return httpx.Response(200, json=_COMPLETED, request=request)


def _error(request: httpx.Request, code: int, message: str) -> httpx.Response:
    return httpx.Response(
        code,
        text='{"type":"error","error":{"type":"rate_limit_error","message":' + json.dumps(message) + "}}",
        headers=MappingProxyType({"retry-after": "0"}),
        request=request,
    )


@contextmanager
def _transport(upstream: Callable[[httpx.Request], httpx.Response]) -> Generator[None]:
    with respx.mock() as transport:
        transport.post("https://api.anthropic.com/v1/messages").mock(side_effect=upstream)
        yield


class _NativeOptions(TypedDict):
    api_key: NotRequired[ReadOnly[str]]
    num_retries: NotRequired[ReadOnly[int]]


async def _call(
    target: Router | None,
    logging_obj: Logging,
    *,
    trusted: bool = True,
    messages: str = _MESSAGES_JSON,
    explicit_logging: bool = True,
) -> None:
    invoke: Final = target.anthropic_messages if target else litellm.anthropic_messages  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy native call signatures
    options: Final = _NativeOptions() if target else _NativeOptions(api_key="test-selected", num_retries=0)
    response: Final[object] = await invoke(  # pyright: ignore[reportUnknownVariableType]  # native Router returns an opaque SDK result
        model="test-router" if target else "anthropic/claude-sonnet-5",
        max_tokens=16,
        stream=_stream(logging_obj),
        messages=_MESSAGES.validate_json(messages),
        **options,
        **_kwargs(logging_obj, trusted, explicit_logging=explicit_logging),
    )
    assert response is not None
    if _stream(logging_obj):
        assert isinstance(response, AsyncIterator)
        stream: Final = cast(AsyncIterator[object], response)  # cast-ok: iterator checked; all items satisfy object
        assert tuple([chunk async for chunk in stream])


async def _reservation(estimator: BaselineCacheEstimator, request_id: str) -> BaselineReservation:
    reservation: Final = estimator.prepare(
        caller_key_hash="test-caller-hash",
        session_id="baseline-session",
        router_id="test-router",
        baseline_deployment_id="baseline",
        target=NativePredictionTarget("claude-opus-5", "test-selected", "https://api.anthropic.com"),
        request_id=request_id,
    )
    assert isinstance(reservation, BaselineReservation)
    assert await estimator.reserve(reservation) is None
    return reservation


def _wire(stream: bool = False) -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=MappingProxyType({"x-api-key": "test-selected"}),
        content=f'{{"model":"claude-sonnet-5","stream":{json.dumps(stream)},"messages":{_MESSAGES_JSON}}}',
    )


def _stamp_native_wire(logging_obj: Logging, clock: _Clock, *, complete: bool = True) -> None:
    timestamp: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # native Logging timing is naive
    logging_obj.completion_start_time = timestamp  # rebind-ok: supply native provider timing evidence
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # legacy native evidence dictionary
        httpx_response=_upstream(_wire(_stream(logging_obj))),
        api_call_start_time=timestamp,
        completion_start_time=timestamp,
        custom_llm_provider="anthropic",
        response_cost=0.125,
        stream=_stream(logging_obj),
        prompt_cache_response_complete=complete,
    )


async def _native_logging(
    estimator: BaselineCacheEstimator, clock: _Clock, request_id: str = "first", *, stream: bool = False
) -> Logging:
    logging_obj: Final = _logging(request_id, stream)
    logging_obj.baseline_cache_context = BaselineCacheContext(estimator, await _reservation(estimator, request_id))
    _stamp_native_wire(logging_obj, clock)
    return logging_obj


async def _finish(estimator: BaselineCacheEstimator, clock: _Clock, request_id: str = "following") -> Logging:
    logging_obj: Final = await _native_logging(estimator, clock, request_id)
    await finalize_baseline_cache(logging_obj, ModelResponse())
    return logging_obj


async def _recover(estimator: BaselineCacheEstimator, clock: _Clock, initial_time: float) -> None:
    boundary: Final = max((clock.now, *(repair.observed_at for repair in estimator.repairs.values())))
    assert boundary - initial_time < 60
    clock.now = round(boundary + 1, 6)  # rebind-ok: advance the injected clock past the actual emergency boundary
    assert (await _finish(estimator, clock)).baseline_cache_estimate == unknown_estimate("history_unavailable")
    clock.now += 1  # rebind-ok: the matching request follows the newly established prefix
    estimate: Final = (await _finish(estimator, clock, "matching")).baseline_cache_estimate
    assert estimate is not None and estimate.cache_read_input_tokens == 5000


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("trusted_stamp", (True, False))
async def test_native_dispatch_reserves_before_upstream_and_stamps_before_callbacks(
    monkeypatch: pytest.MonkeyPatch, stream: bool, trusted_stamp: bool
) -> None:
    rig: Final = _rig(monkeypatch)
    with _transport(_upstream):
        for request_id in ("first", "following"):
            await _call(None, _logging(request_id, stream), trusted=trusted_stamp, explicit_logging=False)
            payload = await rig.capture.payload()
            estimate = _OBJECTS.validate_python(payload["autorouter_savings_estimate"])
            if not trusted_stamp or request_id == "first":
                assert estimate["status"] == "unknown" and payload["autorouter_savings"] is None
                if trusted_stamp:
                    assert estimate["reason"] == "history_unavailable"
            else:
                assert estimate["status"] == "estimated" and estimate["cache_read_input_tokens"] == 5000
                assert estimate["cache_creation_1h_input_tokens"] == 0
                saving = payload["autorouter_savings"]
                assert isinstance(saving, float) and saving < 0


@pytest.mark.parametrize(
    "stream,thinking,invalidation_fails",
    ((False, False, False), (True, False, False), (False, True, False), (False, True, True)),
)
async def test_native_router_retry_with_shared_logging_is_unknown_and_cannot_warm_baseline(
    monkeypatch: pytest.MonkeyPatch, stream: bool, thinking: bool, invalidation_fails: bool
) -> None:
    clock: Final = _Clock()
    rig: Final = _rig(monkeypatch, clock if thinking else None, retries=1)
    requests: Final[list[httpx.Request]] = []  # mutable-ok: compare ordered native attempts and repaired bodies

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) != 1:
            return _upstream(request)
        clock.failures_remaining = 2 if invalidation_fails else 0
        return (
            _error(request, 400, "messages.1.content.0: Invalid `signature` in `thinking` block")
            if thinking
            else _error(request, 429, "retry")
        )

    shared: Final = _logging(stream=stream)
    with _transport(upstream):
        await _call(rig.router, shared, messages=_THINKING_MESSAGES if thinking else _MESSAGES_JSON)
        payload: Final = await rig.capture.payload()
        assert len(requests) == 2 and rig.capture.attempts == ((shared,) if thinking else (shared, shared))
        assert payload["autorouter_savings"] is None and shared.baseline_cache_context is None
        assert _OBJECTS.validate_python(payload["autorouter_savings_estimate"])["reason"] == (
            "estimator_unavailable" if invalidation_fails else "retried_request"
        )
        if thinking:
            assert b'"signature": "invalid"' in requests[0].content and b'"signature"' not in requests[1].content
        else:
            await _call(rig.router, _logging("following", stream))
            following: Final = await rig.capture.payload()
            assert len(requests) == 3 and following["autorouter_savings"] is None
            assert _OBJECTS.validate_python(following["autorouter_savings_estimate"])["reason"] == "history_unavailable"


@pytest.mark.parametrize("defer_clock_fails", (False, True))
async def test_finalize_failure_invalidates_pending_reservation_and_preserves_spend_logging(
    monkeypatch: pytest.MonkeyPatch, defer_clock_fails: bool
) -> None:
    clock: Final = _Clock(round(time.time(), 6))
    rig: Final = _rig(monkeypatch, clock)
    logging_obj: Final = await _native_logging(rig.estimator, clock)
    clock.failures_remaining = 2 if defer_clock_fails else 1
    await logging_obj.async_success_handler(result=ModelResponse(model="claude-sonnet-5"))
    assert (await rig.capture.payload())["response_cost"] == 0.125
    assert logging_obj.baseline_cache_estimate == unknown_estimate("estimator_unavailable")
    assert logging_obj.baseline_cache_context is None
    await _recover(rig.estimator, clock, clock.now)


@pytest.mark.parametrize(
    "operation,retry_owns_reservation,fault",
    (
        *(("finalize", retry, fault) for retry in (False, True) for fault in ("exception", "before", "after")),
        ("finalize", True, "none"),
        ("cancel", False, "none"),
        ("cancel", False, "exception"),
    ),
)
async def test_late_finalize_failure_cleans_original_reservation_without_overwriting_replacement(
    operation: str, retry_owns_reservation: bool, fault: Literal["exception", "before", "after", "none"]
) -> None:
    clock: Final = _Clock()
    gate: Final = _CountingGate()
    estimator: Final = _DelayedCancelEstimator(clock, gate)
    store: Final = _FailingHistoryStore()
    estimator.store = store
    logging_obj: Final = await _native_logging(estimator, clock)
    prepared: Final = (
        None
        if retry_owns_reservation
        else BaselineCacheContext(estimator, await _reservation(estimator, "replacement"))
    )
    finalizing: Final = asyncio.create_task(
        cancel_baseline_cache(logging_obj)
        if operation == "cancel"
        else logging_obj._prepare_baseline_cache_estimate(ModelResponse())  # pyright: ignore[reportPrivateUsage]  # production Logging delegate
    )
    await asyncio.wait_for(gate.started.wait(), timeout=5)
    if retry_owns_reservation:
        await invalidate_baseline_cache(logging_obj, "retried_request")
    replacement: Final = logging_obj.baseline_cache_context if retry_owns_reservation else prepared
    assert replacement is not None
    logging_obj.baseline_cache_context = replacement
    sentinel: Final = (
        None
        if operation == "cancel"
        else unknown_estimate("retried_request" if fault == "none" else "replacement_estimate")
    )
    logging_obj.baseline_cache_estimate = sentinel
    clock.failures_remaining = (2 if operation == "cancel" else 1) if fault == "exception" else 0
    store.fail_next_exchange = fault if fault in ("before", "after") else None
    gate.release.set()
    assert await asyncio.wait_for(finalizing, timeout=5) is (False if operation == "cancel" else None)
    assert logging_obj.baseline_cache_context is replacement and logging_obj.baseline_cache_estimate is sentinel
    if operation == "cancel":
        _stamp_native_wire(logging_obj, clock)
        await finalize_baseline_cache(logging_obj, ModelResponse())
        assert logging_obj.baseline_cache_estimate == unknown_estimate("history_unavailable")
        return
    clock.now = 1001.0
    if retry_owns_reservation:
        assert (await _finish(estimator, clock)).baseline_cache_estimate == unknown_estimate("pending_request")
        clock.now = 1200.0
        await finalize_baseline_cache(logging_obj, ModelResponse())
        assert logging_obj.baseline_cache_context is None
        if fault == "none":
            clock.now = 4601.0
            assert (await _finish(estimator, clock, "matching")).baseline_cache_estimate == unknown_estimate(
                "history_unavailable"
            )
    else:
        assert await estimator.finalize(
            replacement.reservation, wire=_wire(), request_started_at=clock.now, available_at=clock.now
        ) == unknown_estimate("history_unavailable")


@pytest.mark.parametrize("partial_usage", (False, True))
async def test_native_provider_error_and_failure_telemetry_survive_invalidation_fault(
    monkeypatch: pytest.MonkeyPatch, partial_usage: bool
) -> None:
    clock: Final = _Clock()
    rig: Final = _rig(monkeypatch, clock)
    logging_obj: Final = await _native_logging(rig.estimator, clock) if partial_usage else _logging()

    def upstream(request: httpx.Request) -> httpx.Response:
        clock.failures_remaining = 2
        return _error(request, 401, "original upstream authentication failure")

    original: Final[Exception]
    if partial_usage:
        logging_obj.record_partial_usage_for_failure(
            Usage(prompt_tokens=6000, completion_tokens=10), response_cost=0.125
        )
        interrupted: Final = httpx.ReadError("original interrupted stream")
        clock.failures_remaining = 2
        await logging_obj.dispatch_failure_handlers(interrupted, str(interrupted), prefer_async_handlers=True)
        original = interrupted
    else:
        with _transport(upstream):
            with pytest.raises(litellm.AuthenticationError, match="original upstream authentication failure") as caught:
                await _call(None, logging_obj)
        assert caught.value.status_code == 401
        original = caught.value  # pyright: ignore[reportGeneralTypeIssues]  # mutually exclusive request paths bind once
    payload, logged = await asyncio.wait_for(rig.capture.failures.get(), timeout=5)
    assert logged is original and payload["status"] == "failure" and rig.capture.failures.empty()
    assert logging_obj.baseline_cache_estimate == unknown_estimate("estimator_unavailable")
    assert logging_obj.baseline_cache_context is not None and logging_obj.baseline_cache_context.invalidated
    if partial_usage:
        assert (payload["response_cost"], payload["prompt_tokens"], payload["completion_tokens"]) == (0.125, 6000, 10)


@pytest.mark.parametrize("phase", ("setup", "prepare", "reserve", "register_storage"))
async def test_native_generation_and_success_telemetry_survive_predispatch_estimator_fault(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    clock: Final = _Clock()
    clock.failure_at = 1 if phase == "prepare" else 2 if phase == "reserve" else None
    rig: Final = _rig(monkeypatch, clock, lookup_fails=phase == "setup")
    store: Final = _FailingHistoryStore()
    rig.estimator.store = store
    store.fail_next_exchange = "before" if phase == "register_storage" else None
    logging_obj: Final = _logging()
    requests: Final[list[httpx.Request]] = []  # mutable-ok: verify actual dispatch despite estimation faults

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if phase in ("reserve", "register_storage"):
            assert logging_obj.baseline_cache_context is not None and logging_obj.baseline_cache_context.invalidated
        return _upstream(request)

    with _transport(upstream):
        await _call(None, logging_obj)
        payload: Final = await rig.capture.payload()
    cost: Final = payload["response_cost"]
    assert len(requests) == 1 and payload["status"] == "success" and isinstance(cost, float) and cost > 0
    assert logging_obj.baseline_cache_estimate == unknown_estimate(
        "state_unavailable" if phase == "register_storage" else "estimator_unavailable"
    )
    assert logging_obj.baseline_cache_context is None and rig.capture.failures.empty()


async def _wait_for_context(logging_obj: Logging, *, invalidated: bool = False) -> BaselineCacheContext:
    while (
        logging_obj.baseline_cache_context is None or logging_obj.baseline_cache_context.invalidated is not invalidated
    ):
        await asyncio.sleep(0)
    return logging_obj.baseline_cache_context


@pytest.mark.parametrize("mode", ("cancel", "retry", "cancel_retry", "invalidate"))
async def test_late_registration_cannot_cancel_retry_owned_reservation(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    clock: Final = _Clock()
    rig: Final = _rig(monkeypatch, clock)
    logging_obj: Final = await _native_logging(rig.estimator, clock) if mode == "invalidate" else _logging()
    kwargs: Final = _kwargs(logging_obj)
    await rig.estimator.store.lock.acquire()
    active: Final = asyncio.create_task(
        invalidate_baseline_cache(logging_obj, "retried_request")
        if mode == "invalidate"
        else rig.hook.async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages)
    )
    original: Final = await asyncio.wait_for(
        _wait_for_context(logging_obj, invalidated=mode == "invalidate"), timeout=5
    )
    retry: Final = (
        asyncio.create_task(rig.hook.async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages))
        if mode in ("retry", "cancel_retry")
        else None
    )
    retained: Final = (
        await asyncio.wait_for(_wait_for_context(logging_obj, invalidated=True), timeout=5) if retry else original
    )
    assert retained.reservation is original.reservation
    if mode != "retry":
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
    rig.estimator.store.lock.release()
    if mode == "retry":
        await asyncio.wait_for(active, timeout=5)
    if retry:
        await asyncio.wait_for(retry, timeout=5)
    if mode == "cancel":
        assert logging_obj.baseline_cache_context is None
        assert (await _finish(rig.estimator, clock)).baseline_cache_estimate == unknown_estimate("history_unavailable")
        return
    current: Final = logging_obj.baseline_cache_context
    assert current is not None and current.invalidated
    if retry:
        assert current is retained and logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
    assert (await _finish(rig.estimator, clock)).baseline_cache_estimate == unknown_estimate("pending_request")
    clock.now = 1001.0
    _stamp_native_wire(logging_obj, clock)
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_context is None
    assert (await _finish(rig.estimator, clock, "matching")).baseline_cache_estimate == unknown_estimate(
        "history_unavailable"
    )


@pytest.mark.parametrize("wire,completed", ((False, False), (False, True), (True, False)))
async def test_native_stream_logging_without_wire_retains_scope_until_terminal_event(
    monkeypatch: pytest.MonkeyPatch, wire: bool, completed: bool
) -> None:
    clock: Final = _Clock()
    rig: Final = _rig(monkeypatch, clock)
    logging_obj: Final = await _native_logging(rig.estimator, clock, stream=True)
    if not wire:
        logging_obj.model_call_details.pop("httpx_response")  # pyright: ignore[reportUnknownMemberType]  # remove only the observed native wire
    timestamp: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # native timing uses naive timestamps
    await PassThroughStreamingHandler._route_streaming_logging_to_handler(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # actual complete/partial stream logging route
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route="/v1/messages",
        request_body=_JSON_OBJECT.validate_json('{"model":"claude-sonnet-5","stream":true}'),
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=timestamp,
        raw_bytes=_sse(completed),
        end_time=timestamp,
    )
    assert (await rig.capture.payload())["status"] == "success"
    assert logging_obj.baseline_cache_estimate == unknown_estimate(
        "incomplete_response" if wire else "missing_final_wire"
    )
    if completed:
        assert logging_obj.baseline_cache_context is None
    else:
        assert logging_obj.baseline_cache_context is not None and logging_obj.baseline_cache_context.invalidated
        assert (await _finish(rig.estimator, clock)).baseline_cache_estimate == unknown_estimate("pending_request")
    if wire:
        await logging_obj.invalidate_baseline_cache_estimate("retried_request")
        clock.now = 1200.0
        _stamp_native_wire(logging_obj, clock)
        await finalize_baseline_cache(logging_obj, ModelResponse())
        assert logging_obj.baseline_cache_context is None
        assert logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
        clock.now = 4601.0
        assert (await _finish(rig.estimator, clock, "matching")).baseline_cache_estimate == unknown_estimate(
            "history_unavailable"
        )


@pytest.mark.parametrize("retries", (0, 1))
@pytest.mark.parametrize("cleanup_fails", (False, True))
async def test_terminal_proxy_failure_retires_reservation_after_router_exhaustion(
    monkeypatch: pytest.MonkeyPatch, retries: int, cleanup_fails: bool
) -> None:
    clock: Final = _Clock(round(time.time(), 6))
    rig: Final = _rig(monkeypatch, clock, retries)
    shared: Final = _logging()
    requests: Final[list[httpx.Request]] = []  # mutable-ok: verify actual exhausted retry count

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _error(request, 429, "terminal provider failure")

    with _transport(upstream):
        with pytest.raises(litellm.RateLimitError, match="terminal provider failure") as caught:
            await _call(rig.router, shared)
    assert len(requests) == retries + 1 and shared.baseline_cache_context is not None
    proxy: Final = ProxyLogging(UserApiKeyCache())
    proxy.alert_types = []  # mutable-ok: isolate optional alert sinks
    data: Final = _OBJECTS.validate_python(
        MappingProxyType({"model": "test-router", "litellm_call_id": "first", "litellm_logging_obj": shared})
    )
    clock.failures_remaining = 2 if cleanup_fails else 0
    transformed: Final = await proxy.post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # actual proxy terminal owner
        request_data=data,
        original_exception=caught.value,
        user_api_key_dict=UserAPIKeyAuth(request_route="/v1/messages"),
    )
    assert transformed is rig.capture.replacement_error
    assert await asyncio.wait_for(rig.capture.terminal_errors.get(), timeout=5) is caught.value
    assert "litellm_logging_obj" not in data and shared.baseline_cache_context is None
    await _recover(rig.estimator, clock, clock.now)
