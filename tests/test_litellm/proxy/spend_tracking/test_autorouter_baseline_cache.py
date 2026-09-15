import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from redis.exceptions import RedisError

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimate,
    BaselineCacheEstimator,
    BaselineReservation,
    _HistoryStore,  # pyright: ignore[reportPrivateUsage]  # inject faults at the authoritative store boundary
    _Snapshot,  # pyright: ignore[reportPrivateUsage]  # inspect the persisted reservation lifecycle
    _State,  # pyright: ignore[reportPrivateUsage]  # preserve the typed atomic state transition in the fault store
    _StoreFailure,  # pyright: ignore[reportPrivateUsage]  # inject storage failures
)

_MODEL: Final = "claude-sonnet-5"
_TARGET: Final = NativePredictionTarget(_MODEL, "test-provider-key", "https://configured-native-provider.test")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
pytestmark: Final = [pytest.mark.usefixtures("local_model_cost_map"), pytest.mark.asyncio]


@dataclass
class _Clock:
    now: float = 10000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class _Counter:
    unavailable: bool = False
    calls: int = 0

    async def __call__(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        self.calls += 1
        return None if self.unavailable else json.dumps(_JSON_OBJECT.validate_python(body)).count("token ")


def _block(tokens: int = 0, ttl: str | None = None, *, text: str = "") -> dict[str, JsonValue]:
    block: Final[dict[str, JsonValue]] = {"type": "text", "text": text or "token " * tokens}
    if ttl is not None:
        block["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    return block


def _native(*blocks: dict[str, JsonValue], target: NativePredictionTarget = _TARGET) -> httpx.Request:
    return httpx.Request(
        "POST", f"{target.api_base}/v1/messages",
        headers=MappingProxyType({"anthropic-version": "2023-06-01", "x-api-key": target.api_key}),
        json={"model": _MODEL, "max_tokens": 10, "messages": [{"role": "user", "content": list(blocks)}]},
    )


def _wire(
    ttl: str = "1h", *, growth: int = 0, changed: bool = False, target: NativePredictionTarget = _TARGET
) -> httpx.Request:
    blocks: Final = [_block(text=("changed " if changed else "") + "token " * 6000)]
    if growth:
        blocks.append(_block(growth))
    return _native(*blocks, _block(ttl=ttl, text="end"), target=target)


class _FaultStore(_HistoryStore):
    def __init__(self) -> None:
        super().__init__(DualCache())
        self.fault: Literal["read", "before", "after", "conflict"] | None = None
        self.remaining = 0
        self.after_write: Callable[[], None] | None = None

    def arm(self, fault: Literal["read", "before", "after", "conflict"]) -> None:
        self.fault = fault
        self.remaining = 4 if fault == "conflict" else 1

    async def read(self, scope: str, now: float) -> _Snapshot | _StoreFailure:
        if self.fault == "read" and self.remaining:
            self.remaining -= 1
            return _StoreFailure()
        return await super().read(scope, now)

    async def exchange(self, scope: str, before: _Snapshot, after: _State, now: float) -> bool | _StoreFailure:
        if self.remaining and self.fault in ("before", "conflict"):
            self.remaining -= 1
            return _StoreFailure() if self.fault == "before" else False
        applied: Final = await super().exchange(scope, before, after, now)
        callback: Final = self.after_write
        if callback is not None:
            self.after_write = None
            callback()
        if self.remaining and self.fault == "after":
            self.remaining -= 1
            return _StoreFailure()
        return applied


@dataclass
class _Rig:
    clock: _Clock = field(default_factory=_Clock)
    counter: _Counter = field(default_factory=_Counter)
    faults: _FaultStore = field(default_factory=_FaultStore)
    estimator: BaselineCacheEstimator = field(init=False)

    def __post_init__(self) -> None:
        self.estimator = BaselineCacheEstimator(DualCache(), self.clock, self.counter)
        self.estimator.store = self.faults

    def prepare(
        self, request: str, *, caller: str = "caller", session: str = "session",
        target: NativePredictionTarget = _TARGET,
    ) -> BaselineReservation:
        reservation: Final = self.estimator.prepare(
            caller_key_hash=caller, session_id=session, router_id="router",
            baseline_deployment_id="baseline", target=target, request_id=request,
        )
        assert isinstance(reservation, BaselineReservation)
        return reservation

    async def reserve(self, request: str) -> BaselineReservation:
        reservation: Final = self.prepare(request)
        assert await self.estimator.reserve(reservation) is None
        return reservation

    async def finish(
        self, reservation: BaselineReservation, wire: httpx.Request | None = None,
        *, available_at: float | None = None,
    ) -> BaselineCacheEstimate:
        return await self.estimator.finalize(
            reservation, wire=wire if wire is not None else _wire(), request_started_at=reservation.reserved_at,
            available_at=self.clock.now if available_at is None else available_at,
        )

    async def run(self, request: str, wire: httpx.Request | None = None) -> BaselineCacheEstimate:
        reservation: Final = await self.reserve(request)
        self.clock.advance(0.1)
        return await self.finish(reservation, wire)

    async def state(self, reservation: BaselineReservation) -> _State:
        snapshot: Final = await self.estimator.store.read(reservation.scope, self.clock.now)
        assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
        return snapshot.state

    def fork(self) -> "_Rig":
        other: Final = _Rig(clock=self.clock)
        other.estimator.store = self.estimator.store
        return other


@pytest.fixture
def rig() -> _Rig:
    return _Rig()


@pytest.mark.parametrize(
    "ttl,duration,bucket",
    (("5m", 300, "cache_creation_5m_input_tokens"), ("1h", 3600, "cache_creation_1h_input_tokens")),
)
async def test_established_expiry_never_receives_a_hypothetical_read(
    rig: _Rig, ttl: str, duration: int, bucket: str
) -> None:
    first: Final = await rig.run("first", _wire(ttl))
    assert (first.status, first.reason) == ("unknown", "history_unavailable")
    rig.clock.now = 10001.0
    warm: Final = await rig.run("warm", _wire(ttl))
    assert warm.cache_read_input_tokens == 6000
    assert warm.cache_creation_1h_input_tokens == warm.cache_creation_5m_input_tokens == 0
    rig.clock.now = 10001.0 + duration
    expired: Final = await rig.run("expired", _wire(ttl))
    assert (expired.status, expired.reason) == ("estimated", "cache_prefix_expired")
    assert expired.cache_read_input_tokens == 0
    assert expired.metadata()[bucket] == 6000
    usage: Final = expired.usage(10)
    assert usage is not None and usage.prompt_tokens == 6000 and usage.total_tokens == 6010
    assert rig.counter.calls <= 4


@pytest.mark.parametrize("ttl,duration", (("5m", 300), ("1h", 3600)))
@pytest.mark.parametrize("cancelled", (False, True))
async def test_invalidation_keeps_unknown_cache_effects_through_latest_retry_completion(
    rig: _Rig, ttl: str, duration: int, cancelled: bool
) -> None:
    await rig.run("seed", _wire(ttl))
    rig.clock.now = 10001.0
    assert (await rig.run("warm", _wire(ttl))).cache_read_input_tokens == 6000
    rig.clock.now = 10005.0
    failed: Final = await rig.reserve("retrying")
    if cancelled:
        await rig.estimator.cancel(failed)
    calls: Final = rig.counter.calls
    for at, reason in ((10010.0, "upstream_request_failed"), (10010.0 + duration, "retried_upstream_request")):
        rig.clock.now = at
        invalidated: BaselineCacheEstimate = await rig.estimator.invalidate(failed, reason)
        assert (invalidated.status, invalidated.reason) == ("unknown", reason)
    assert rig.counter.calls == calls
    rig.clock.now = 10020.0 + duration
    following: Final = await rig.run("following", _wire(ttl))
    assert (following.status, following.reason) == ("unknown", "history_unavailable")
    assert following.cache_read_input_tokens is None
    rig.clock.now = 10020.0 + 2 * duration
    expired: Final = await rig.run("expired", _wire(ttl))
    assert (expired.status, expired.reason) == ("estimated", "cache_prefix_expired")
    assert expired.cache_read_input_tokens == 0


async def test_prefix_change_is_cold_after_history_horizon_and_unknown_before_it(rig: _Rig) -> None:
    await rig.run("first")
    assert (await rig.run("changed-early", _wire(changed=True))).status == "unknown"
    rig.clock.now = 13601.0
    changed: Final = await rig.run("changed", _wire(growth=2000))
    assert changed.status == "estimated"
    assert changed.cache_read_input_tokens == 0
    assert changed.cache_creation_1h_input_tokens == 8000


async def test_lookback_reads_prior_marker_and_writes_only_growth(rig: _Rig) -> None:
    await rig.run("first")
    rig.clock.now = 13600.0
    await rig.run("cold")
    growing_wire: Final = _native(_block(6000), _block(text="end"), _block(2000, "1h"))
    grown: Final = await rig.run("grown", growing_wire)
    assert grown.status == "estimated"
    assert grown.cache_read_input_tokens == 6000
    assert grown.cache_creation_1h_input_tokens == 2000
    rig.clock.now = 17200.05
    assert (await rig.run("old-prefix")).cache_read_input_tokens == 6000


async def test_mixed_ttl_keeps_new_hour_and_five_minute_writes_separate(rig: _Rig) -> None:
    await rig.run("seed")
    rig.clock.now = 13600.0
    wire: Final = _native(_block(5000, "1h"), _block(2000, "5m"), _block(100))
    result: Final = await rig.run("mixed", wire)
    assert result.status == "estimated"
    assert result.input_tokens == 100
    assert result.cache_read_input_tokens == 0
    assert result.cache_creation_1h_input_tokens == 5000
    assert result.cache_creation_5m_input_tokens == 2000


async def test_parallel_pending_requests_and_late_completions_do_not_self_hit_or_regress_refresh(rig: _Rig) -> None:
    first: Final = await rig.reserve("first")
    rig.clock.now = 10001.0
    second: Final = await rig.reserve("second")
    rig.clock.now = 10002.0
    second_result: Final = await rig.finish(second)
    assert second_result.reason == "pending_request"
    assert (await rig.finish(first)).cache_read_input_tokens is None
    assert (await rig.finish(second, _wire(changed=True))) == second_result
    rig.clock.now = 13600.5
    assert (await rig.run("still-warm")).cache_read_input_tokens == 6000


@pytest.mark.parametrize("complete,cache_hit", ((False, False), (True, True)))
async def test_failed_incomplete_or_gateway_cache_responses_do_not_warm_state(
    rig: _Rig, complete: bool, cache_hit: bool
) -> None:
    reservation: Final = await rig.reserve("ignored")
    result: Final = await rig.estimator.finalize(
        reservation, wire=_wire(), request_started_at=rig.clock.now, available_at=rig.clock.now,
        completed=complete, cache_hit=cache_hit,
    )
    assert result.status == "unknown"
    assert (await rig.run("following")).reason == "history_unavailable"


async def test_provider_counts_are_memoized_and_failures_remain_unknown(rig: _Rig) -> None:
    rig.counter.unavailable = True
    unavailable: Final = await rig.run("unavailable")
    assert unavailable.reason == "token_count_unavailable"
    assert unavailable.usage(5) is None
    rig.counter.unavailable = False
    assert (await rig.run("first")).reason == "history_unavailable"
    calls: Final = rig.counter.calls
    assert (await rig.run("second")).cache_read_input_tokens == 6000
    assert rig.counter.calls == calls


@pytest.mark.parametrize(
    "caller,session,target",
    (
        ("other", "session", _TARGET),
        ("caller", "other", _TARGET),
        ("caller", "session", NativePredictionTarget(_MODEL, "other-key", _TARGET.api_base)),
        ("caller", "session", NativePredictionTarget(_MODEL, "test-provider-key", "https://other.test")),
    ),
)
async def test_scope_isolates_callers_sessions_and_baseline_credentials(
    rig: _Rig, caller: str, session: str, target: NativePredictionTarget
) -> None:
    await rig.run("first")
    isolated: Final = rig.prepare("second", caller=caller, session=session, target=target)
    assert await rig.estimator.reserve(isolated) is None
    assert (await rig.finish(isolated, _wire(target=target))).reason == "history_unavailable"


@pytest.mark.parametrize("baseline_base,wire_base,wire_key,supported", (
    ("https://gateway.example", "https://gateway.example", _TARGET.api_key, True),
    ("https://gateway.example/v1/messages", "https://gateway.example", _TARGET.api_key, True),
    ("https://api.anthropic.com", "https://api.anthropic.com", _TARGET.api_key, True),
    ("https://other.example", "https://gateway.example", _TARGET.api_key, False),
    ("https://gateway.example/other", "https://gateway.example", _TARGET.api_key, False),
    ("https://api.anthropic.com", "https://gateway.example", _TARGET.api_key, False),
    ("https://gateway.example", "https://gateway.example", "other-account", False),
    ("https://gateway.example", "https://gateway.example", None, False),
), ids=("gateway-root", "messages-path", "first-party", "other-host", "other-path",
        "gateway-to-first-party", "other-key", "missing-key"))
async def test_baseline_count_uses_only_the_captured_request_recipient(
    rig: _Rig, baseline_base: str, wire_base: str, wire_key: str | None, supported: bool
) -> None:
    target: Final = NativePredictionTarget("claude-opus-5", _TARGET.api_key, baseline_base)
    reservation: Final = rig.prepare("recipient", target=target)
    assert await rig.estimator.reserve(reservation) is None
    wire: Final = httpx.Request(
        "POST", wire_base + "/v1/messages", content=_native(_block(100, "1h")).content,
        headers={"x-api-key": wire_key} if wire_key is not None else {},
    )
    result: Final = await rig.finish(reservation, wire)
    assert result.status == ("estimated" if supported else "unknown")
    assert (rig.counter.calls > 0) is supported
    state: Final = await rig.state(reservation)
    assert not state.pending and not state.versions
    if not supported:
        assert result.reason == "unsupported_baseline_recipient"


@pytest.mark.parametrize(
    "tokens,ttl,observed,reason,expected_input",
    (
        (1, "1h", 0, "below_cache_minimum", 1),
        (6000, None, 0, "no_cache_breakpoints", 6000),
        (6000, None, 6000, "implicit_cache_without_breakpoints", None),
    ),
    ids=("below-minimum", "unmarked-ordinary", "unmarked-provider-cache"),
)
async def test_uncacheable_and_implicit_cache_inputs(
    rig: _Rig, tokens: int, ttl: str | None, observed: int, reason: str, expected_input: int | None
) -> None:
    reservation: Final = await rig.reserve("uncacheable")
    result: Final = await rig.estimator.finalize(
        reservation, wire=_native(_block(tokens, ttl)), request_started_at=rig.clock.now,
        available_at=rig.clock.now, observed_cache_tokens=observed,
    )
    assert result.reason == reason and result.input_tokens == expected_input
    if observed:
        assert result.status == "unknown" and rig.counter.calls == 0
    else:
        assert result.status == "estimated"
        assert result.cache_read_input_tokens == result.cache_creation_1h_input_tokens == 0


async def test_ttl_changes_remain_unknown_until_every_possible_refreshed_entry_expires(rig: _Rig) -> None:
    await rig.run("first", _wire("1h"))
    assert (await rig.run("warm", _wire("1h"))).cache_read_input_tokens == 6000
    for request, at in (("changed", 10002.0), ("ambiguous", 10303.0), ("refreshed", 13602.5)):
        rig.clock.now = at
        changed: BaselineCacheEstimate = await rig.run(request, _wire("5m"))
        assert (changed.status, changed.reason) == ("unknown", "cache_ttl_changed")
    rig.clock.now = 17204.0
    expired: Final = await rig.run("expired", _wire("5m"))
    assert expired.status == "estimated"
    assert expired.cache_read_input_tokens == 0
    assert expired.cache_creation_5m_input_tokens == 6000


async def test_pruned_pending_reservation_cannot_advance_cache_state(rig: _Rig) -> None:
    first: Final = await rig.reserve("first")
    for index in range(256):
        await rig.reserve(f"pending-{index}")
    assert (await rig.finish(first)).reason == "reservation_unavailable"


async def test_long_running_request_keeps_the_cache_history_needed_at_its_start(rig: _Rig) -> None:
    await rig.run("seed", _wire("5m"))
    rig.clock.now = 10001.0
    delayed: Final = await rig.reserve("delayed")
    rig.clock.now = 15000.0
    result: Final = await rig.finish(delayed, _wire("5m"))
    assert result.status == "estimated"
    assert result.cache_read_input_tokens == 6000


async def test_redis_is_authoritative_and_faults_never_fall_back_to_local_warmth(rig: _Rig) -> None:
    backend: Final = RedisCache(host="127.0.0.1", port=6398, namespace=f"baseline-state-test-{id(rig)}")
    try:
        await backend.async_delete_cache("unused")
    except (RedisError, OSError):
        pytest.skip("isolated integration Redis is unavailable")
    rig.estimator = BaselineCacheEstimator(DualCache(redis_cache=backend), rig.clock, rig.counter)
    second: Final = _Rig(clock=rig.clock)
    second.estimator = BaselineCacheEstimator(DualCache(redis_cache=backend), rig.clock, second.counter)
    first: Final = await rig.reserve("first")
    rig.clock.advance(0.1)
    await rig.finish(first)
    assert (await second.run("second")).cache_read_input_tokens == 6000
    await backend.async_delete_cache(first.scope)
    assert (await rig.run("lost")).reason == "history_unavailable"
    rig.estimator = BaselineCacheEstimator(
        DualCache(redis_cache=RedisCache(host="127.0.0.1", port=1)), rig.clock, rig.counter
    )
    refused: Final = await rig.estimator.reserve(rig.prepare("broken"))
    assert isinstance(refused, BaselineCacheEstimate) and refused.reason == "state_unavailable"


@pytest.mark.parametrize("operation", ("reserve-terminal", "reserve-cancel", "finalize", "invalidate", "cancel"))
@pytest.mark.parametrize("fault", ("read", "before", "after", "conflict"))
async def test_storage_failure_retires_exact_request_on_recovery(
    rig: _Rig, operation: str, fault: Literal["read", "before", "after", "conflict"]
) -> None:
    await rig.run("seed")
    failed: Final = rig.prepare("failed")
    if not operation.startswith("reserve-"):
        assert await rig.estimator.reserve(failed) is None
    rig.faults.arm(fault)
    if operation.startswith("reserve-"):
        result: Final = await rig.estimator.reserve(failed)
        assert result is not None and result.status == "unknown"
        if operation == "reserve-cancel":
            await rig.estimator.cancel(failed)
        else:
            assert (await rig.run("during-unacknowledged")).reason == "pending_request"
            rig.clock.advance(10)
            await rig.estimator.invalidate(failed, "registration_failed", completed=True)
    elif operation == "finalize":
        assert (await rig.finish(failed)).status == "unknown"
    elif operation == "invalidate":
        assert (await rig.estimator.invalidate(failed, "final_response_unknown")).status == "unknown"
    else:
        await rig.estimator.cancel(failed)
    recovered: Final = await rig.run("recovered")
    if operation in ("cancel", "reserve-cancel"):
        assert recovered.cache_read_input_tokens == 6000
    else:
        assert recovered.reason == "history_unavailable"
    state: Final = await rig.state(failed)
    assert not any(item.request_id == failed.request_id for item in state.pending)
    assert any(item.request_id == failed.request_id for item in state.completed)


@pytest.mark.parametrize("ordering", ("durable-cross-process", "active-persisted", "finish-applied", "finish-deferred"))
async def test_failed_old_finalizer_preserves_newer_retry_in_every_atomic_order(rig: _Rig, ordering: str) -> None:
    await rig.run("seed")
    retrying: Final = await rig.reserve("retry-race")
    observer: Final = rig.fork() if ordering == "durable-cross-process" else rig
    if ordering == "durable-cross-process":
        rig.faults.arm("after")
        await rig.estimator.invalidate(retrying, "retry_active", completed=False)
    elif ordering == "active-persisted":
        await rig.estimator.invalidate(retrying, "retry_active", completed=False)
        rig.faults.arm("read")
    elif ordering == "finish-applied":
        def activate_during_ack() -> None:
            rig.estimator.defer(retrying, "retry_active", completed=False)

        rig.faults.after_write = activate_during_ack
        rig.faults.arm("after")
    else:
        rig.faults.arm("before")
    failed: Final = await observer.finish(retrying)
    assert failed.status == "unknown"
    if ordering == "durable-cross-process":
        assert failed.reason == "retry_active"
    if ordering == "finish-deferred":
        rig.estimator.defer(retrying, "retry_active", completed=False)
    assert (await observer.run("during-retry")).reason == "pending_request"
    state: Final = await observer.state(retrying)
    own: Final = next(item for item in state.pending if item.request_id == retrying.request_id)
    assert own.invalidated_reason == "retry_active"
    rig.clock.advance(10)
    await rig.estimator.invalidate(retrying, "retry_terminal", completed=True)
    assert (await observer.run("after-retry-terminal")).reason == "history_unavailable"


@pytest.mark.parametrize("unsupported", (False, True))
async def test_active_invalidation_supersedes_acknowledged_provisional_finish_but_not_logical_terminal(
    rig: _Rig, unsupported: bool,
) -> None:
    completed: Final = await rig.reserve("provisional")
    valid: Final = _wire()
    wire: Final = httpx.Request("POST", valid.url, headers=valid.headers, content="invalid") if unsupported else valid
    await rig.finish(completed, wire)
    await rig.estimator.invalidate(completed, "new_retry", completed=False)
    assert (await rig.run("during-new-retry")).reason == "pending_request"
    await rig.estimator.invalidate(completed, "logical_terminal", completed=True)
    await rig.estimator.invalidate(completed, "late_failure_callback", completed=False)
    state: Final = await rig.state(completed)
    assert not any(item.request_id == completed.request_id for item in state.pending)
    final: Final = next(item for item in state.completed if item.request_id == completed.request_id)
    assert final.terminal and final.estimate.status == "unknown"


async def test_acknowledged_older_repair_cannot_erase_new_terminal_debt(rig: _Rig) -> None:
    active: Final = await rig.reserve("overlap")
    rig.estimator.defer(active, "still_active", completed=False)

    def complete_while_acknowledging() -> None:
        rig.clock.advance(10)
        rig.estimator.defer(active, "now_terminal", completed=True)

    rig.faults.after_write = complete_while_acknowledging
    intermediate: Final = await rig.estimator.reserve(active)
    assert intermediate is not None and intermediate.reason == "still_active"
    assert (await rig.run("after-overlap")).reason == "history_unavailable"
    state: Final = await rig.state(active)
    assert not any(item.request_id == active.request_id for item in state.pending)
    completed: Final = next(item for item in state.completed if item.request_id == active.request_id)
    assert completed.estimate.reason == "now_terminal"
    assert state.uncertain_before >= 10010.0


@pytest.mark.parametrize("early_result", ("pending", "completed"))
async def test_idempotent_results_publish_other_cleanup_without_removing_live_requests(
    rig: _Rig, early_result: str
) -> None:
    existing: Final = await rig.reserve("existing")
    if early_result == "completed":
        await rig.finish(existing)
    abandoned: Final = await rig.reserve("abandoned")
    unrelated: Final = await rig.reserve("unrelated-live")
    rig.estimator.defer(abandoned, "abandoned", completed=True)
    if early_result == "pending":
        assert await rig.estimator.reserve(existing) is None
    else:
        duplicate: Final = await rig.finish(existing)
        assert duplicate.status == "unknown"
    state: Final = await rig.state(existing)
    pending_ids: Final = frozenset(item.request_id for item in state.pending)
    assert abandoned.request_id not in pending_ids
    assert unrelated.request_id in pending_ids
    assert (existing.request_id in pending_ids) == (early_result == "pending")


async def test_terminal_repairs_survive_one_cache_ttl_and_late_active_callbacks_cannot_resurrect_pending(
    rig: _Rig
) -> None:
    ended: Final = await rig.reserve("ended")
    rig.estimator.defer(ended, "terminal", completed=True)
    rig.clock.advance(3601)
    other_scope: Final = rig.prepare("other-scope", session="other")
    rig.estimator.defer_cancel(other_scope)
    await rig.run("recover-after-hour")
    await rig.estimator.invalidate(ended, "late_active_callback", completed=False)
    assert not any(item.request_id == ended.request_id for item in (await rig.state(ended)).pending)
    assert (await rig.run("after-late-callback")).reason == "history_unavailable"


@pytest.mark.parametrize("loss", ("restart", "repair-overflow"))
async def test_lost_unpublished_repairs_preserve_pending_uncertainty(rig: _Rig, loss: str) -> None:
    pending: Final = await rig.reserve("lost-terminal")
    if loss == "restart":
        rig.faults.arm("before")
        await rig.estimator.invalidate(pending, "terminal")
    else:
        rig.estimator.defer(pending, "lost_repair", completed=True)
        for index in range(1024):
            rig.estimator.defer(rig.prepare(f"overflow-{index}", session=f"scope-{index}"), "uncertain", completed=True)
        assert len(rig.estimator.repairs) == 1024
    observer: Final = rig.fork() if loss == "restart" else rig
    assert (await observer.run("after-repair-loss")).reason == "pending_request"
    rig.clock.advance(7201)
    assert (await observer.run("after-retention")).reason == "history_unavailable"


@pytest.mark.parametrize("cancel_last", (False, True))
async def test_cancel_and_invalidation_debt_order_never_resurrects_pending(rig: _Rig, cancel_last: bool) -> None:
    await rig.run("seed")
    reservation: Final = await rig.reserve("ordered")
    if cancel_last:
        rig.estimator.defer(reservation, "registration_unknown", completed=False)
        rig.estimator.defer_cancel(reservation)
    else:
        rig.estimator.defer_cancel(reservation)
        rig.estimator.defer(reservation, "late_upstream_evidence", completed=False)
    following: Final = await rig.run("after-ordered")
    assert following.reason != "pending_request"
    if cancel_last:
        assert following.cache_read_input_tokens == 6000
    else:
        assert following.reason == "history_unavailable"


@pytest.mark.parametrize("publish_fault_first", (False, True), ids=("success-first", "fault-first"))
@pytest.mark.parametrize("available_at", (10001.0, 10002.0, 10003.0), ids=("before", "equal", "after"))
async def test_independent_success_cannot_cross_fault_cutoff_by_finalizer_order(
    rig: _Rig, publish_fault_first: bool, available_at: float
) -> None:
    invalidator: Final = rig.fork()
    seed: Final = await rig.reserve("delayed-independent-success")
    rig.clock.now = 10001.0
    failed: Final = await invalidator.reserve("terminal-failure")
    rig.clock.now = 10002.0
    invalidator.estimator.defer(failed, "failed_request", completed=True)
    if publish_fault_first:
        assert await invalidator.estimator.reserve(failed) is not None
    rig.clock.now = 10004.0
    await rig.finish(seed, available_at=available_at)
    if not publish_fault_first:
        assert await invalidator.estimator.reserve(failed) is not None
    state: Final = await rig.state(seed)
    assert state.invalidated_before == 10002.0 and not state.pending
    assert next(item for item in state.completed if item.request_id == failed.request_id).terminal
    if available_at > 10002.0:
        assert len(state.versions) == 1 and state.versions[0].expires_at == 13600.0
    else:
        assert not state.versions
    rig.clock.now = 10005.0
    following: Final = await rig.run("after-terminal-and-independent-success")
    if available_at > 10002.0:
        assert following.cache_read_input_tokens == 6000
    else:
        assert following.reason == "history_unavailable"
    assert (await rig.run("matching-after-restored-evidence")).cache_read_input_tokens == 6000


@pytest.mark.parametrize("history,cutoff", (("new", None), ("legacy", None), ("overflow", 10002.0)))
async def test_history_initialization_and_cancel_preserve_only_explicit_fault_cutoffs(
    rig: _Rig, history: str, cutoff: float | None
) -> None:
    if history == "legacy":
        assert _State.model_validate_json('{"uncertain_before":10000.0}').invalidated_before is None
        return
    if history == "overflow":
        rig.clock.now = 10004.0
        rig.estimator.uncertainty_floor = 10002.0
    initial: Final = await rig.reserve("initial")
    if history == "new":
        await rig.finish(initial)
    cancelled: Final = await rig.reserve("never-dispatched") if history == "new" else initial
    await rig.estimator.cancel(cancelled)
    state: Final = await rig.state(initial)
    assert state.invalidated_before == cutoff
    assert state.uncertain_before == rig.clock.now
    if history == "new":
        assert (await rig.run("matches-initial-history-timestamp")).cache_read_input_tokens == 6000


async def test_persisted_pre_cutoff_version_is_removed_before_matching(rig: _Rig) -> None:
    initial: Final = await rig.reserve("old-persisted-success")
    await rig.finish(initial)
    snapshot: Final = await rig.estimator.store.read(initial.scope, rig.clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    cutoff_state: Final = snapshot.state.model_copy(update=MappingProxyType({"invalidated_before": 10001.0}))
    assert await rig.estimator.store.exchange(initial.scope, snapshot, cutoff_state, rig.clock.now) is True
    rig.clock.now = 10002.0
    assert (await rig.run("must-not-read-persisted-old-evidence")).reason == "history_unavailable"
    assert (await rig.run("fresh-evidence-after-persisted-cutoff")).cache_read_input_tokens == 6000
