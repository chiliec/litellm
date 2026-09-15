from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import Final, Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.caching.dual_cache import DualCache
from litellm.llms.anthropic.prompt_cache_prediction import (
    CountedBreakpoint,
    CountedPromptCachePlan,
    NativePredictionTarget,
    TokenCounter,
    UnsupportedCachePlan,
    count_cache_plan,
    count_prompt_tokens,
    parse_cache_plan,
    supported_baseline_recipient,
    supported_prediction_headers,
)
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage
from litellm.utils import get_prompt_cache_min_tokens

_MAX_TTL: Final = 3600
_RETENTION_SECONDS: Final = 86400
_MAX_REQUESTS: Final = 256
_MAX_VERSIONS: Final = 1024
_MAX_SCOPES: Final = 1024
_MAX_COUNTS: Final = 4096
_MAX_REPAIRS: Final = 1024
_COUNT_TIMEOUT: Final = 3.0
_STORE_TIMEOUT: Final = 1.0
_CAS_ATTEMPTS: Final = 4
_JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])
_GET_SCRIPT: Final = "return redis.call('GET', KEYS[1])"
_CAS_SCRIPT: Final = """
local current = redis.call('GET', KEYS[1])
if (current or '') ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 1
"""


class BaselineCacheEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    status: Literal["estimated", "unknown"]
    reason: str
    input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_5m_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_1h_input_tokens: int | None = Field(default=None, ge=0)

    def usage(self, completion_tokens: int) -> Usage | None:
        if (
            self.status != "estimated"
            or self.input_tokens is None
            or self.cache_read_input_tokens is None
            or self.cache_creation_5m_input_tokens is None
            or self.cache_creation_1h_input_tokens is None
        ):
            return None
        writes: Final = self.cache_creation_5m_input_tokens + self.cache_creation_1h_input_tokens
        total: Final = self.input_tokens + self.cache_read_input_tokens + writes
        return Usage(
            prompt_tokens=total,
            completion_tokens=completion_tokens,
            total_tokens=total + completion_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=self.input_tokens,
                cached_tokens=self.cache_read_input_tokens,
                cache_creation_tokens=writes,
                cache_write_tokens=writes,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=self.cache_creation_5m_input_tokens,
                    ephemeral_1h_input_tokens=self.cache_creation_1h_input_tokens,
                ),
            ),
        )

    def metadata(self) -> dict[str, JsonValue]:  # mutable-ok: spend-log JSON serializers require a plain dictionary
        return _JSON_BODY.validate_python(self.model_dump(mode="json", exclude_none=True))


def unknown_estimate(reason: str) -> BaselineCacheEstimate:
    return BaselineCacheEstimate(status="unknown", reason=reason)


@dataclass(frozen=True, slots=True)
class BaselineReservation:
    scope: str
    request_id: str
    reserved_at: float
    target: NativePredictionTarget


class _Pending(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request_id: str
    reserved_at: float
    invalidated_reason: str | None = None


class _Completed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request_id: str
    completed_at: float
    estimate: BaselineCacheEstimate
    terminal: bool = False


class _Version(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    fingerprint: str
    content_fingerprint: str
    prefix_tokens: int = Field(ge=0)
    ttl_seconds: Literal[300, 3600]
    started_at: float
    available_at: float
    expires_at: float
    uncertain: bool = False


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    uncertain_before: float
    invalidated_before: float | None = None
    pending: tuple[_Pending, ...] = ()
    completed: tuple[_Completed, ...] = ()
    versions: tuple[_Version, ...] = ()


@dataclass(frozen=True, slots=True)
class _Repair:
    scope: str
    request_id: str
    reserved_at: float
    observed_at: float
    reason: str | None
    kind: Literal["active", "terminal", "cancel", "finish"]


@dataclass(frozen=True, slots=True)
class _Update:
    state: _State
    estimate: BaselineCacheEstimate | None


def _repair_state(state: _State, repair: _Repair) -> _State:
    prior: Final = next((item for item in state.completed if item.request_id == repair.request_id), None)
    pending: Final = next((item for item in state.pending if item.request_id == repair.request_id), None)
    active: Final = pending is not None and pending.invalidated_reason is not None
    reason: Final = (
        pending.invalidated_reason if repair.kind == "finish" and pending is not None and active else repair.reason
    )
    final: Final = (prior is not None and prior.terminal) or repair.kind in ("terminal", "cancel")
    terminal: Final = final or (repair.kind == "finish" and not active)
    remaining: Final = tuple(item for item in state.pending if item.request_id != repair.request_id)
    estimate: Final = unknown_estimate(reason or "request_cancelled")
    return _State(
        uncertain_before=max(state.uncertain_before, repair.observed_at) if reason else state.uncertain_before,
        invalidated_before=(
            max(state.invalidated_before or 0.0, repair.observed_at) if reason else state.invalidated_before
        ),
        pending=(
            remaining
            if terminal
            else (
                *remaining,
                _Pending(
                    request_id=repair.request_id,
                    reserved_at=pending.reserved_at if pending is not None else repair.reserved_at,
                    invalidated_reason=reason,
                ),
            )
        ),
        completed=(
            (
                *(item for item in state.completed if item.request_id != repair.request_id),
                _Completed(
                    request_id=repair.request_id,
                    completed_at=max(repair.observed_at, prior.completed_at if prior is not None else 0.0),
                    estimate=prior.estimate if prior is not None and prior.terminal and reason is None else estimate,
                    terminal=final,
                ),
            )
            if terminal
            else tuple(item for item in state.completed if item.request_id != repair.request_id)
        ),
        versions=state.versions,
    )


@dataclass(frozen=True, slots=True)
class _Snapshot:
    raw: str
    state: _State | None


@dataclass(frozen=True, slots=True)
class _StoreFailure:
    reason: str = "state_unavailable"


@runtime_checkable
class _Script(Protocol):
    def __call__(self, *, keys: Sequence[str], args: Sequence[str | bytes | int | float]) -> Awaitable[object]: ...


def _script(value: object) -> _Script | None:
    return value if isinstance(value, _Script) else None


class _HistoryStore:
    def __init__(self, cache: DualCache) -> None:
        self.cache = cache
        self.local: Mapping[str, tuple[float, str]] = MappingProxyType({})
        self.lock = asyncio.Lock()

    async def read(self, scope: str, now: float) -> _Snapshot | _StoreFailure:
        if self.cache.redis_cache is None:
            local_value: Final = self.local.get(scope)
            raw: Final = local_value[1] if local_value is not None and local_value[0] > now else None
            return self._decode(raw)
        try:
            script: Final = _script(self.cache.redis_cache.async_register_script(_GET_SCRIPT))
            if script is None:
                return _StoreFailure()
            value: Final[object] = await asyncio.wait_for(script(keys=(scope,), args=()), timeout=_STORE_TIMEOUT)
            return self._decode(value)
        except Exception:  # noqa: BLE001  # storage faults must not establish cache absence
            return _StoreFailure()

    @staticmethod
    def _decode(raw: object) -> _Snapshot | _StoreFailure:
        if raw is None:
            return _Snapshot(raw="", state=None)
        if not isinstance(raw, (str, bytes)):
            return _StoreFailure()
        try:
            text: Final = raw.decode() if isinstance(raw, bytes) else raw
            return _Snapshot(raw=text, state=_State.model_validate_json(text))
        except (UnicodeDecodeError, ValidationError):
            return _StoreFailure()

    async def exchange(self, scope: str, before: _Snapshot, after: _State, now: float) -> bool | _StoreFailure:
        serialized: Final = after.model_dump_json()
        if self.cache.redis_cache is not None:
            try:
                script: Final = _script(self.cache.redis_cache.async_register_script(_CAS_SCRIPT))
                if script is None:
                    return _StoreFailure()
                result: Final[object] = await asyncio.wait_for(
                    script(keys=(scope,), args=(before.raw, serialized, _RETENTION_SECONDS)),
                    timeout=_STORE_TIMEOUT,
                )
            except Exception:  # noqa: BLE001  # preserve uncertainty when Redis rejects an atomic update
                return _StoreFailure()
            return result == 1
        async with self.lock:
            existing: Final = self.local.get(scope)
            current: Final = existing[1] if existing is not None and existing[0] > now else ""
            if current != before.raw:
                return False
            remaining: Final = tuple(
                (key, entry) for key, entry in self.local.items() if key != scope and entry[0] > now
            )[-(_MAX_SCOPES - 1) :]
            self.local = MappingProxyType(
                {key: entry for key, entry in (*remaining, (scope, (now + _RETENTION_SECONDS, serialized)))}
            )
            return True


@dataclass(frozen=True, slots=True)
class _Count:
    tokens: int
    expires_at: float


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _key(caller: str, session: str, router: str, deployment: str, target: NativePredictionTarget) -> str:
    return "autorouter-baseline-cache:v1:" + _digest(
        (caller, session, router, deployment, target.model, target.api_key, target.api_base)
    )


def _bounded(state: _State, now: float) -> _State:
    stale_pending: Final = tuple(item for item in state.pending if item.reserved_at < now - 2 * _MAX_TTL)
    pending: Final = tuple(item for item in state.pending if item.reserved_at >= now - 2 * _MAX_TTL)
    needed_since: Final = min((now - _MAX_TTL, *(item.reserved_at for item in pending)))
    live_versions: Final = tuple(
        version
        for version in state.versions
        if version.expires_at >= needed_since
        and (state.invalidated_before is None or version.available_at > state.invalidated_before)
    )
    dropped_versions: Final = live_versions[:-_MAX_VERSIONS]
    dropped_pending: Final = pending[:-_MAX_REQUESTS]
    uncertainty: Final = max(
        (
            state.uncertain_before,
            *(version.started_at for version in dropped_versions),
            now if stale_pending or dropped_pending else 0.0,
        )
    )
    return _State(
        uncertain_before=uncertainty,
        invalidated_before=state.invalidated_before,
        pending=pending[-_MAX_REQUESTS:],
        completed=state.completed[-_MAX_REQUESTS:],
        versions=live_versions[-_MAX_VERSIONS:],
    )


def _eligible(plan: CountedPromptCachePlan, minimum: int) -> tuple[CountedBreakpoint, ...]:
    return tuple(marker for marker in plan.breakpoints if marker.prefix_tokens >= minimum)


def _matching(state: _State, markers: tuple[CountedBreakpoint, ...], started: float) -> tuple[_Version, ...]:
    return tuple(
        version
        for version in state.versions
        if not version.uncertain
        and version.available_at <= started < version.expires_at
        and any(
            version.fingerprint in marker.lookback_fingerprints
            and version.prefix_tokens <= marker.prefix_tokens
            and version.ttl_seconds == marker.ttl_seconds
            for marker in markers
        )
    )


def _ambiguous(state: _State, markers: tuple[CountedBreakpoint, ...], started: float) -> tuple[_Version, ...]:
    return tuple(
        version
        for version in state.versions
        if version.available_at <= started < version.expires_at
        and any(
            version.content_fingerprint in marker.lookback_content_fingerprints
            and (version.uncertain or version.ttl_seconds != marker.ttl_seconds)
            for marker in markers
        )
    )


def _estimate(
    state: _State, request_id: str, plan: CountedPromptCachePlan, minimum: int, started: float
) -> BaselineCacheEstimate:
    markers: Final = _eligible(plan, minimum)
    if not markers:
        return BaselineCacheEstimate(
            status="estimated",
            reason="below_cache_minimum" if plan.breakpoints else "no_cache_breakpoints",
            input_tokens=plan.total_tokens,
            cache_read_input_tokens=0,
            cache_creation_5m_input_tokens=0,
            cache_creation_1h_input_tokens=0,
        )
    if any(item.request_id != request_id and item.reserved_at <= started for item in state.pending):
        return unknown_estimate("pending_request")
    if _ambiguous(state, markers, started):
        return unknown_estimate("cache_ttl_changed")
    candidates: Final = _matching(state, markers, started)
    read: Final = max((version.prefix_tokens for version in candidates), default=0)
    end: Final = markers[-1].prefix_tokens
    if read < end and started < state.uncertain_before + max(marker.ttl_seconds for marker in markers):
        return unknown_estimate("history_unavailable")
    one_hour: Final = max(
        (marker.prefix_tokens for marker in markers if marker.ttl_seconds == 3600 and marker.prefix_tokens > read),
        default=read,
    )
    expired: Final = any(
        version.expires_at <= started and any(version.fingerprint in marker.lookback_fingerprints for marker in markers)
        for version in state.versions
    )
    return BaselineCacheEstimate(
        status="estimated",
        reason="cache_prefix_available" if read else "cache_prefix_expired" if expired else "cache_prefix_cold",
        input_tokens=plan.total_tokens - end,
        cache_read_input_tokens=read,
        cache_creation_5m_input_tokens=end - one_hour,
        cache_creation_1h_input_tokens=one_hour - read,
    )


def _new_versions(
    state: _State,
    markers: tuple[CountedBreakpoint, ...],
    started: float,
    available: float,
) -> tuple[_Version, ...]:
    ambiguous: Final = _ambiguous(state, markers, started)
    longest_ttl: Final = max((version.expires_at - version.started_at for version in ambiguous), default=0)
    candidates: Final = () if ambiguous else _matching(state, markers, started)
    hit: Final = max(candidates, key=lambda version: version.prefix_tokens, default=None)
    refresh: Final = (
        (
            _Version(
                fingerprint=hit.fingerprint,
                content_fingerprint=hit.content_fingerprint,
                prefix_tokens=hit.prefix_tokens,
                ttl_seconds=hit.ttl_seconds,
                started_at=started,
                available_at=available,
                expires_at=started + hit.ttl_seconds,
            ),
        )
        if hit is not None and all(marker.fingerprint != hit.fingerprint for marker in markers)
        else ()
    )
    return (
        *refresh,
        *(
            _Version(
                fingerprint=marker.fingerprint,
                content_fingerprint=marker.content_fingerprint,
                prefix_tokens=marker.prefix_tokens,
                ttl_seconds=3600 if marker.ttl_seconds == 3600 else 300,
                started_at=started,
                available_at=available,
                expires_at=started + max(longest_ttl, marker.ttl_seconds),
                uncertain=bool(ambiguous),
            )
            for marker in markers
        ),
    )


class BaselineCacheEstimator:
    def __init__(
        self,
        cache: DualCache,
        clock: Callable[[], float] = time.time,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = _HistoryStore(cache)
        self.clock = clock
        self.token_counter = token_counter
        self.counts: Mapping[str, _Count] = MappingProxyType({})
        self.count_slots = asyncio.Semaphore(8)
        self.repairs: Mapping[tuple[str, str], _Repair] = MappingProxyType({})
        self.uncertainty_floor = 0.0

    def prepare(
        self,
        *,
        caller_key_hash: str,
        session_id: str,
        router_id: str,
        baseline_deployment_id: str,
        target: NativePredictionTarget,
        request_id: str,
    ) -> BaselineReservation | BaselineCacheEstimate:
        if not all((caller_key_hash, session_id, router_id, baseline_deployment_id, request_id)):
            return unknown_estimate("missing_baseline_scope")
        try:
            now: Final = self.clock()
        except Exception:  # noqa: BLE001  # optional preparation cannot prevent generation
            return unknown_estimate("estimator_unavailable")
        if not math.isfinite(now):
            return unknown_estimate("estimator_unavailable")
        return BaselineReservation(
            _key(caller_key_hash, session_id, router_id, baseline_deployment_id, target),
            _digest(request_id),
            now,
            target,
        )

    def _repair_time(self, reservation: BaselineReservation) -> float:
        try:
            now: Final = self.clock()
            if math.isfinite(now):
                return max(now, reservation.reserved_at)
        except Exception:  # noqa: BLE001  # emergency cleanup must survive the original clock fault
            return max(time.time(), reservation.reserved_at)
        return max(time.time(), reservation.reserved_at)

    def _defer(
        self,
        reservation: BaselineReservation,
        reason: str | None,
        kind: Literal["active", "terminal", "cancel", "finish"],
    ) -> None:
        now: Final = self._repair_time(reservation)
        key: Final = (reservation.scope, reservation.request_id)
        prior: Final = self.repairs.get(key)
        repair: Final = _Repair(
            scope=reservation.scope,
            request_id=reservation.request_id,
            reserved_at=reservation.reserved_at,
            observed_at=max(now, prior.observed_at if prior is not None else 0.0),
            reason=prior.reason if kind == "finish" and prior is not None else reason,
            kind=(
                prior.kind
                if kind == "finish" and prior is not None
                else "terminal"
                if kind == "active" and prior is not None and prior.kind in ("terminal", "cancel")
                else kind
            ),
        )
        retained: Final = tuple(
            (identity, item)
            for identity, item in self.repairs.items()
            if identity != key and max(item.reserved_at + 2 * _MAX_TTL, item.observed_at + _MAX_TTL) >= now
        )
        self.uncertainty_floor = max(
            (
                self.uncertainty_floor,
                *(item.observed_at for _, item in retained[: -(_MAX_REPAIRS - 1)] if item.reason is not None),
            )
        )
        self.repairs = MappingProxyType(
            {identity: item for identity, item in (*retained[-(_MAX_REPAIRS - 1) :], (key, repair))}
        )

    def defer(self, reservation: BaselineReservation, reason: str, *, completed: bool) -> None:
        self._defer(reservation, reason, "terminal" if completed else "active")

    def defer_cancel(self, reservation: BaselineReservation) -> None:
        self._defer(reservation, None, "cancel")

    async def _transition(
        self,
        reservation: BaselineReservation,
        operation: Callable[[_State, bool, float], _Update],
    ) -> BaselineCacheEstimate | _StoreFailure | None:
        async def attempt(remaining: int) -> BaselineCacheEstimate | _StoreFailure | None:
            now: Final = self.clock()
            repairs: Final = tuple(item for item in self.repairs.values() if item.scope == reservation.scope)
            floor: Final = self.uncertainty_floor
            before: Final = await self.store.read(reservation.scope, now)
            if isinstance(before, _StoreFailure):
                return before
            loaded: Final = before.state or _State(uncertain_before=now)
            initial: Final = loaded.model_copy(
                update=MappingProxyType(
                    {
                        "uncertain_before": max(loaded.uncertain_before, floor),
                        "invalidated_before": (
                            max(loaded.invalidated_before or 0.0, floor) if floor > 0.0 else loaded.invalidated_before
                        ),
                    }
                )
            )
            state: Final = _bounded(reduce(_repair_state, repairs, initial), now)
            update: Final = operation(state, before.state is None, now)
            after: Final = _bounded(update.state, now)
            exchanged: Final = await self.store.exchange(reservation.scope, before, after, now)
            if isinstance(exchanged, _StoreFailure):
                return exchanged
            if not exchanged:
                return await attempt(remaining - 1) if remaining > 1 else _StoreFailure("state_contention")
            acknowledged: Final = MappingProxyType({(item.scope, item.request_id): item for item in repairs})
            self.repairs = MappingProxyType(
                {identity: item for identity, item in self.repairs.items() if acknowledged.get(identity) is not item}
            )
            return update.estimate

        return await attempt(_CAS_ATTEMPTS)

    async def reserve(self, reservation: BaselineReservation) -> BaselineCacheEstimate | None:
        def register(state: _State, _missing: bool, _now: float) -> _Update:
            finished: Final = next(
                (item for item in state.completed if item.request_id == reservation.request_id), None
            )
            if finished is not None:
                return _Update(state, finished.estimate)
            pending: Final = next((item for item in state.pending if item.request_id == reservation.request_id), None)
            if pending is not None:
                return _Update(
                    state, unknown_estimate(pending.invalidated_reason) if pending.invalidated_reason else None
                )
            return _Update(
                state.model_copy(
                    update=MappingProxyType(
                        {
                            "pending": (
                                *state.pending,
                                _Pending(request_id=reservation.request_id, reserved_at=reservation.reserved_at),
                            )
                        }
                    )
                ),
                None,
            )

        result: Final = await self._transition(reservation, register)
        if isinstance(result, _StoreFailure):
            self.defer(reservation, result.reason, completed=False)
            return unknown_estimate(result.reason)
        return result

    async def _count(self, target: NativePredictionTarget, body: Mapping[str, JsonValue]) -> int | None:
        cache_key: Final = _digest((target.model, target.api_key, target.api_base, _JSON_BODY.validate_python(body)))
        now: Final = self.clock()
        cached: Final = self.counts.get(cache_key)
        if cached is not None and cached.expires_at > now:
            return cached.tokens

        async def execute() -> int | None:
            async with self.count_slots:
                return (
                    await self.token_counter(target.model, target.api_key, body)
                    if self.token_counter is not None
                    else await count_prompt_tokens(target.model, target.api_key, body, api_base=target.api_base)
                )

        try:
            tokens: Final = await asyncio.wait_for(execute(), timeout=_COUNT_TIMEOUT)
        except Exception:  # noqa: BLE001  # provider counting cannot fail a completed generation
            return None
        if tokens is None or tokens < 0:
            return None
        retained: Final = tuple(
            (key, value) for key, value in self.counts.items() if value.expires_at > now and key != cache_key
        )[-(_MAX_COUNTS - 1) :]
        self.counts = MappingProxyType(
            {key: value for key, value in (*retained, (cache_key, _Count(tokens, now + _MAX_TTL)))}
        )
        return tokens

    async def finalize(
        self,
        reservation: BaselineReservation,
        *,
        wire: httpx.Request,
        request_started_at: float,
        available_at: float,
        completed: bool = True,
        cache_hit: bool = False,
        observed_cache_tokens: int = 0,
    ) -> BaselineCacheEstimate:
        if cache_hit:
            await self.cancel(reservation)
            return unknown_estimate("response_cache_hit")
        if not completed:
            return await self.invalidate(reservation, "incomplete_response")
        if not reservation.reserved_at <= request_started_at <= available_at <= self.clock():
            return await self._finish(reservation, None, request_started_at, available_at, "invalid_request_timing")
        if not supported_baseline_recipient(reservation.target, wire):
            return await self._finish(
                reservation, None, request_started_at, available_at, "unsupported_baseline_recipient"
            )
        try:
            body: Final = _JSON_BODY.validate_json(wire.content)
        except (ValidationError, RuntimeError, httpx.RequestNotRead):
            return await self._finish(reservation, None, request_started_at, available_at, "invalid_wire_request")
        if not supported_prediction_headers(wire.headers):
            return await self._finish(
                reservation, None, request_started_at, available_at, "unsupported_request_headers"
            )
        plan: Final = parse_cache_plan(body)
        if isinstance(plan, UnsupportedCachePlan):
            return await self._finish(reservation, None, request_started_at, available_at, plan.reason)
        if not plan.breakpoints and observed_cache_tokens > 0:
            return await self._finish(
                reservation, None, request_started_at, available_at, "implicit_cache_without_breakpoints"
            )

        async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
            return await self._count(reservation.target, body)

        try:
            counted: Final = await asyncio.wait_for(
                count_cache_plan(reservation.target.model, reservation.target.api_key, plan, token_counter=count),
                timeout=_COUNT_TIMEOUT,
            )
        except TimeoutError:
            return await self._finish(reservation, None, request_started_at, available_at, "token_count_timeout")
        return await self._finish(
            reservation,
            None if isinstance(counted, UnsupportedCachePlan) else counted,
            request_started_at,
            available_at,
            counted.reason if isinstance(counted, UnsupportedCachePlan) else None,
        )

    async def _finish(
        self,
        reservation: BaselineReservation,
        plan: CountedPromptCachePlan | None,
        started: float,
        available: float,
        reason: str | None,
    ) -> BaselineCacheEstimate:
        minimum: Final = get_prompt_cache_min_tokens(reservation.target.model)

        def finish(state: _State, missing: bool, now: float) -> _Update:
            prior: Final = next((item for item in state.completed if item.request_id == reservation.request_id), None)
            if prior is not None:
                return _Update(state, prior.estimate)
            pending: Final = next((item for item in state.pending if item.request_id == reservation.request_id), None)
            if pending is not None and pending.invalidated_reason:
                return _Update(state, unknown_estimate(pending.invalidated_reason))
            if plan is None or pending is None:
                unavailable: Final = (
                    "history_unavailable" if missing else "reservation_unavailable" if pending is None else reason
                )
                unknown: Final = unknown_estimate(unavailable or "unsupported_request")
                return _Update(
                    _repair_state(
                        state,
                        _Repair(
                            reservation.scope,
                            reservation.request_id,
                            reservation.reserved_at,
                            now,
                            unknown.reason,
                            "finish",
                        ),
                    ),
                    unknown,
                )
            estimate: Final = _estimate(state, reservation.request_id, plan, minimum, started)
            versions: Final = _new_versions(state, _eligible(plan, minimum), started, available)
            return _Update(
                _State(
                    uncertain_before=state.uncertain_before,
                    invalidated_before=state.invalidated_before,
                    pending=tuple(item for item in state.pending if item.request_id != reservation.request_id),
                    completed=(
                        *state.completed,
                        _Completed(request_id=reservation.request_id, completed_at=now, estimate=estimate),
                    ),
                    versions=(*state.versions, *versions),
                ),
                estimate,
            )

        result: Final = await self._transition(reservation, finish)
        if isinstance(result, _StoreFailure):
            self._defer(reservation, result.reason, "finish")
            return unknown_estimate(result.reason)
        return result if result is not None else unknown_estimate("estimator_unavailable")

    async def invalidate(
        self, reservation: BaselineReservation, reason: str, *, completed: bool = True
    ) -> BaselineCacheEstimate:
        self.defer(reservation, reason, completed=completed)

        def invalidated(state: _State, _missing: bool, _now: float) -> _Update:
            return _Update(state, unknown_estimate(reason))

        result: Final = await self._transition(reservation, invalidated)
        return unknown_estimate(result.reason) if isinstance(result, _StoreFailure) else unknown_estimate(reason)

    async def cancel(self, reservation: BaselineReservation) -> None:
        self.defer_cancel(reservation)

        def cancelled(state: _State, _missing: bool, _now: float) -> _Update:
            return _Update(state, None)

        await self._transition(reservation, cancelled)
