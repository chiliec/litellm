from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # shared legacy metadata owner is validated below
)
from litellm.llms.anthropic.prompt_cache_prediction import (
    UnsupportedPredictionTarget,
    resolve_baseline_prediction_target,
)
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimate,
    BaselineCacheEstimator,
    BaselineReservation,
    unknown_estimate,
)
from litellm.proxy.spend_tracking.savings import (
    _proxy_llm_router,  # pyright: ignore[reportPrivateUsage]  # shared optional proxy-router owner
)
from litellm.types.router import BaselineRouteStamp
from litellm.types.utils import CallTypes, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.utils import InternalUsageCache
    from litellm.router import Router

_METADATA: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class BaselineCacheContext:
    estimator: BaselineCacheEstimator
    reservation: BaselineReservation
    invalidated: bool = False


@contextmanager
def _baseline_cache_operation(
    logging_obj: Logging,
    context: BaselineCacheContext | None,
    *,
    action: Literal["reserve", "cancel", "invalidate", "complete"],
) -> Generator[None]:
    try:
        yield
    except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001  # optional estimation cannot replace inference or logging outcomes
        current: Final = logging_obj.baseline_cache_context
        retry_owned: Final = (
            context is not None
            and current is not None
            and current is not context
            and current.reservation is context.reservation
        )
        cancelled: Final = isinstance(error, asyncio.CancelledError)
        cancel: Final = action == "cancel" or (cancelled and action == "reserve" and not retry_owned)
        if context is not None:
            if cancel and not retry_owned:
                context.estimator.defer_cancel(context.reservation)
            else:
                context.estimator.defer(
                    context.reservation, "estimator_unavailable", completed=action == "complete" and not retry_owned
                )
        if current is context:
            logging_obj.baseline_cache_estimate = unknown_estimate(  # rebind-ok: publish on the captured Logging owner
                "estimator_unavailable"
            )
            logging_obj.baseline_cache_context = (  # rebind-ok: retain only a nonterminal request's retry ownership
                BaselineCacheContext(context.estimator, context.reservation, invalidated=True)
                if context is not None and not cancel and action != "complete"
                else None
            )
        if cancelled:
            raise


class _Metadata(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    route: BaselineRouteStamp = Field(alias="_autorouter_baseline_route")
    user_api_key_hash: str = Field(min_length=1)
    session_id: str | None = None


class _WireEvent(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    httpx_response: httpx.Response
    api_call_start_time: datetime
    completion_start_time: datetime
    custom_llm_provider: str
    stream: bool = False
    prompt_cache_response_complete: bool = False
    cache_hit: bool | None = None


class _ResponseUsage(BaseModel):
    model_config = ConfigDict(strict=True, from_attributes=True)
    usage: Usage | None = None


class AutoRouterBaselineCache(CustomLogger):
    def __init__(
        self,
        internal_usage_cache: InternalUsageCache,
        router: Callable[[], Router | None] = _proxy_llm_router,
        estimator: BaselineCacheEstimator | None = None,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # legacy callback constructor
        self.estimator = estimator if estimator is not None else BaselineCacheEstimator(internal_usage_cache.dual_cache)
        self.router = router

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if not isinstance(logging_obj, Logging) or call_type != CallTypes.anthropic_messages:
            return
        try:
            metadata: Final = _METADATA.validate_python(
                get_litellm_metadata_from_kwargs(
                    {"litellm_params": kwargs}  # mutable-ok: legacy metadata helper requires a dict envelope
                )
            )
        except ValidationError:
            return
        if metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
            return
        if logging_obj.baseline_cache_attempted:
            await logging_obj.invalidate_baseline_cache_estimate("retried_request")
            return
        logging_obj.baseline_cache_attempted = True
        try:
            request: Final = _Metadata.model_validate(metadata)
        except ValidationError:
            return
        await cancel_baseline_cache(logging_obj)
        prepared: Final = self._prepare_context(logging_obj, kwargs, request)
        if isinstance(prepared, BaselineCacheEstimate):
            logging_obj.baseline_cache_estimate = prepared
            return
        if prepared is None:
            return
        context: Final = prepared
        logging_obj.baseline_cache_estimate = None
        logging_obj.baseline_cache_context = context
        with _baseline_cache_operation(logging_obj, context, action="reserve"):
            estimate: Final = await context.estimator.reserve(context.reservation)
            current: Final = logging_obj.baseline_cache_context
            if current is not context:
                if current is None or current.reservation is not context.reservation:  # pyright: ignore[reportUnnecessaryComparison]  # another callback can release ownership during registration
                    await cancel_baseline_cache(logging_obj, expected_context=context)
                return
            if estimate is not None:
                logging_obj.baseline_cache_estimate = estimate
                logging_obj.baseline_cache_context = (
                    BaselineCacheContext(  # rebind-ok: preserve registration uncertainty through completion
                        context.estimator, context.reservation, invalidated=True
                    )
                )

    def _prepare_context(
        self, logging_obj: Logging, kwargs: Mapping[str, object], request: _Metadata
    ) -> BaselineCacheContext | BaselineCacheEstimate | None:
        with _baseline_cache_operation(logging_obj, None, action="reserve"):
            session: Final = kwargs.get("litellm_session_id") or request.session_id or logging_obj.litellm_session_id
            if not isinstance(session, str) or not session:
                return unknown_estimate("missing_session")
            router: Final = self.router()
            deployment: Final = router.get_deployment(request.route.baseline_deployment_id) if router else None
            if deployment is None:
                return unknown_estimate("missing_baseline_deployment")
            target: Final = resolve_baseline_prediction_target(deployment.litellm_params)
            if isinstance(target, UnsupportedPredictionTarget):
                return unknown_estimate(target.reason)
            reservation: Final = self.estimator.prepare(
                caller_key_hash=request.user_api_key_hash,
                session_id=session,
                router_id=request.route.router_name,
                baseline_deployment_id=request.route.baseline_deployment_id,
                target=target,
                request_id=logging_obj.litellm_call_id,
            )
            if isinstance(reservation, BaselineCacheEstimate):
                return reservation
            return BaselineCacheContext(self.estimator, reservation)
        return None


async def cancel_baseline_cache(logging_obj: Logging, *, expected_context: BaselineCacheContext | None = None) -> bool:
    context: Final = expected_context if expected_context is not None else logging_obj.baseline_cache_context
    if context is None:
        return False
    with _baseline_cache_operation(logging_obj, context, action="cancel"):
        await context.estimator.cancel(context.reservation)
        if logging_obj.baseline_cache_context is not context:
            return False
        logging_obj.baseline_cache_context = None  # rebind-ok: release this logging request's reservation
        return True
    return False


async def invalidate_baseline_cache(
    logging_obj: Logging,
    reason: str,
    *,
    completed: bool = False,
    expected_context: BaselineCacheContext | None = None,
) -> None:
    context: Final = expected_context if expected_context is not None else logging_obj.baseline_cache_context
    if context is None:
        logging_obj.baseline_cache_estimate = unknown_estimate(reason)  # rebind-ok: no reservation remains to abandon
        return
    invalidated: Final = BaselineCacheContext(context.estimator, context.reservation, invalidated=True)
    if logging_obj.baseline_cache_context is context:
        logging_obj.baseline_cache_context = invalidated  # rebind-ok: older callbacks must not own this invalidation
        logging_obj.baseline_cache_estimate = unknown_estimate(reason)  # rebind-ok: invalidate before awaiting storage
    with _baseline_cache_operation(logging_obj, invalidated, action="complete" if completed else "invalidate"):
        estimate: Final = await context.estimator.invalidate(context.reservation, reason, completed=completed)
        if logging_obj.baseline_cache_context is not invalidated:
            return
        logging_obj.baseline_cache_estimate = estimate  # rebind-ok: preserve a storage failure's unknown result
        if completed:
            logging_obj.baseline_cache_context = None  # rebind-ok: terminal completion releases retained retry scope


async def finalize_baseline_cache(logging_obj: Logging, response_obj: object) -> None:
    context: Final = logging_obj.baseline_cache_context
    if context is None:
        return
    event_details: Final = _METADATA.validate_python(logging_obj.model_call_details)
    if event_details.get("cache_hit") is True:
        if await cancel_baseline_cache(logging_obj):
            logging_obj.baseline_cache_estimate = unknown_estimate("response_cache_hit")  # rebind-ok: no upstream call
        return
    try:
        event: Final = _WireEvent.model_validate(logging_obj.model_call_details)
        wire: Final = event.httpx_response.request
    except (ValidationError, RuntimeError):
        await invalidate_baseline_cache(
            logging_obj,
            "missing_final_wire",
            completed=event_details.get("stream") is False
            or event_details.get("prompt_cache_response_complete") is True,
            expected_context=context,
        )
        return
    complete: Final = (
        event.custom_llm_provider == "anthropic"
        and event.httpx_response.status_code == 200
        and (not event.stream or event.prompt_cache_response_complete)
    )
    if context.invalidated or not complete:
        prior: Final = logging_obj.baseline_cache_estimate
        reason: Final = prior.reason if context.invalidated and prior is not None else "incomplete_response"
        await invalidate_baseline_cache(logging_obj, reason, completed=complete, expected_context=context)
        return
    if logging_obj.baseline_cache_estimate is not None:
        return
    with _baseline_cache_operation(logging_obj, context, action="complete"):
        usage: Final = (
            _ResponseUsage.model_validate(response_obj).usage if isinstance(response_obj, ModelResponse) else None
        )
        details: Final = usage.prompt_tokens_details if isinstance(usage, Usage) else None
        observed_cache_tokens: Final = (
            (details.cached_tokens or 0) + (details.cache_creation_tokens or details.cache_write_tokens or 0)
            if details is not None
            else 0
        )
        estimate: Final = await context.estimator.finalize(
            context.reservation,
            wire=wire,
            request_started_at=event.api_call_start_time.timestamp(),
            available_at=event.completion_start_time.timestamp(),
            completed=complete,
            cache_hit=event.cache_hit is True,
            observed_cache_tokens=observed_cache_tokens,
        )
        if logging_obj.baseline_cache_context is not context:
            return
        logging_obj.baseline_cache_estimate = estimate  # rebind-ok: stamp only the context that completed
        logging_obj.baseline_cache_context = None  # rebind-ok: finalize consumes this logging request's reservation
