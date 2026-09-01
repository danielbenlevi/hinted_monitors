"""Provider-aware, reproducible monitor-model construction.

The final monitor experiment intentionally uses several API providers through
one Inspect interface.  This module is the sole place where mutable aliases,
provider routing, reasoning controls, and narrow compatibility shims are
resolved.  Keeping that policy here prevents the three monitor stages from
silently issuing different requests for the same configured model.
"""

from __future__ import annotations

import contextvars
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping


PINNED_OPENAI_MODELS = {
    "openai/gpt-5.4-mini": "openai/gpt-5.4-mini-2026-03-17",
    "openai/gpt-4.1-mini": "openai/gpt-4.1-mini-2025-04-14",
}
MONITOR_MODEL_ALIASES = {
    # Preserve compatibility with the previously documented native-Google ID,
    # but route final Gemini monitor calls through OpenRouter.
    "google/gemini-3.6-flash": "openrouter/google/gemini-3.6-flash",
    "google/gemini-2.5-flash-lite": "openrouter/google/gemini-2.5-flash-lite",
}
OPENROUTER_UPSTREAMS = {
    "anthropic/claude-sonnet-5": "anthropic",
    "anthropic/claude-haiku-4.5": "anthropic",
    "google/gemini-3.6-flash": "google-ai-studio",
    "google/gemini-2.5-flash-lite": "google-ai-studio",
    # Use Z.AI's first-party OpenRouter endpoint exclusively. Other providers
    # serving this open-weight model may use quantized deployments.
    "z-ai/glm-4.5-air": "z-ai",
    # Keep the dated DeepSeek revision on Fireworks' OpenRouter endpoint so the
    # final experiment cannot silently switch inference deployments.
    "deepseek/deepseek-v4-flash-0731": "fireworks",
    # Pin the provider, rather than an endpoint variant. OpenRouter's base
    # provider slug remains stable across endpoint/quantization migrations.
    "qwen/qwen3.6-flash": "alibaba",
    # MiniMax M2.5 uses mandatory native reasoning. Lock final runs to
    # MiniMax's first-party endpoint instead of another hosted deployment.
    "minimax/minimax-m2.5": "minimax",
}
LEGACY_OPENROUTER_UPSTREAM_ALIASES = {
    "qwen/qwen3.6-flash": {"alibaba/fp8": "alibaba"},
}
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
DEFAULT_REASONING_EFFORTS = {
    # GLM-4.5-Air is a hybrid-thinking model. Its final-experiment profile uses
    # reasoning unless the caller explicitly supplies another effort.
    "openrouter/z-ai/glm-4.5-air": "high",
    # OpenRouter exposes low/high/max for the dated 0731 revision and reports
    # high as its default. Materialize that choice in experiment attestations.
    "openrouter/deepseek/deepseek-v4-flash-0731": "high",
}
LOCKED_OPENROUTER_UPSTREAM_LABELS = {
    "z-ai/glm-4.5-air": "official Z.AI",
    "deepseek/deepseek-v4-flash-0731": "Fireworks",
    "minimax/minimax-m2.5": "official MiniMax",
}
ZERO_THINKING_MODELS = {
    # Gemini 2.5 otherwise uses a dynamic thinking budget when Inspect receives
    # no explicit budget. Keep the deliberately weak final configuration stable.
    "openrouter/google/gemini-2.5-flash-lite",
}
MAX_OUTPUT_TOKENS_BY_MODEL = {
    # Successful action-only calls in the frozen run peaked below 5k tokens;
    # bound pathological repetition without changing the prompt or score parser.
    "openrouter/google/gemini-2.5-flash-lite": 8192,
}
SYSTEM_ONLY_USER_ROLE_MODELS = {
    # Z.AI rejects a chat request whose only message has the system role
    # (provider error 1214: "messages parameter is illegal"). MonitoringBench
    # supplies its complete prompt as one system message, so preserve the exact
    # content while representing that otherwise standalone message as user.
    "openrouter/z-ai/glm-4.5-air",
    # Fireworks returns an empty one-token completion for this DeepSeek
    # revision when the complete MonitoringBench prompt is a standalone system
    # message. Preserve the exact content while using the accepted user role.
    "openrouter/deepseek/deepseek-v4-flash-0731",
    # MiniMax's first-party OpenRouter endpoint likewise returns an immediate
    # empty completion when MonitoringBench supplies its complete prompt as a
    # standalone system message. Preserve the prompt content and represent the
    # otherwise standalone message using the accepted user role.
    "openrouter/minimax/minimax-m2.5",
}


@dataclass(frozen=True)
class MonitorModelSpec:
    """Immutable score-affecting model request specification."""

    requested_model: str
    model: str
    api_provider: str
    upstream_provider: str
    reasoning_effort: str | None
    reasoning_tokens: int | None
    requested_temperature: float | None
    effective_temperature: float | None
    openrouter_routing: Mapping[str, Any] | None
    tool_choice_policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "model": self.model,
            "api_provider": self.api_provider,
            "upstream_provider": self.upstream_provider,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_tokens": self.reasoning_tokens,
            "requested_temperature": self.requested_temperature,
            "effective_temperature": self.effective_temperature,
            "openrouter_routing": dict(self.openrouter_routing or {}),
            "tool_choice_policy": self.tool_choice_policy,
        }

    def model_args(self) -> dict[str, Any]:
        if self.api_provider != "openrouter":
            return {}
        return {
            "provider": dict(self.openrouter_routing or {}),
            # OpenRouter includes routing attempts/resolution in the raw response
            # retained by Inspect's ModelCall transcript when this header is set.
            "default_headers": {"X-OpenRouter-Metadata": "enabled"},
        }


def normalize_monitor_model_spec_for_resume(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Canonicalize retired OpenRouter endpoint aliases for paid-output reuse.

    This is deliberately model-specific. It recognizes only a provider endpoint
    rename and does not make different models or upstream providers compatible.
    """

    normalized = json.loads(json.dumps(dict(spec or {}), default=str))
    model = str(normalized.get("model") or "")
    provider_model = model.removeprefix("openrouter/").casefold()
    aliases = LEGACY_OPENROUTER_UPSTREAM_ALIASES.get(provider_model, {})
    if not aliases:
        return normalized
    upstream = normalized.get("upstream_provider")
    if isinstance(upstream, str):
        normalized["upstream_provider"] = aliases.get(upstream.casefold(), upstream)
    routing = normalized.get("openrouter_routing")
    if isinstance(routing, dict):
        for field in ("only", "order", "ignore"):
            values = routing.get(field)
            if isinstance(values, list):
                routing[field] = [
                    aliases.get(value.casefold(), value) if isinstance(value, str) else value
                    for value in values
                ]
    return normalized


def monitor_model_spec_resume_compatible(
    recorded_spec: Mapping[str, Any] | None,
    expected_spec: MonitorModelSpec,
) -> bool:
    return normalize_monitor_model_spec_for_resume(recorded_spec) == normalize_monitor_model_spec_for_resume(
        expected_spec.as_dict()
    )


def resolve_monitor_model_spec(
    model: str,
    *,
    reasoning_effort: str | None,
    reasoning_tokens: int | None = None,
    temperature: float | None = 0.0,
    openrouter_provider: str | None = None,
    apply_monitor_transport_aliases: bool = True,
) -> MonitorModelSpec:
    requested_model = str(model).strip()
    if not requested_model or "/" not in requested_model:
        raise ValueError("Monitor model must be an Inspect provider-qualified model ID.")
    if ":free" in requested_model or "latest" in requested_model.casefold():
        raise ValueError("Mutable/free model aliases are not permitted in the reproducible final protocol.")
    if reasoning_effort not in (None, *REASONING_EFFORTS):
        raise ValueError(f"Unsupported reasoning effort: {reasoning_effort!r}.")
    if reasoning_tokens is not None and reasoning_tokens < 1:
        zero_budget_model = MONITOR_MODEL_ALIASES.get(
            requested_model,
            PINNED_OPENAI_MODELS.get(requested_model, requested_model),
        ).casefold()
        if reasoning_tokens != 0 or zero_budget_model not in ZERO_THINKING_MODELS:
            raise ValueError("--reasoning-tokens must be a positive integer.")
    if reasoning_effort is not None and reasoning_tokens is not None:
        raise ValueError("Choose either reasoning effort or an explicit reasoning-token budget, not both.")

    canonical_model = PINNED_OPENAI_MODELS.get(requested_model, requested_model)
    if apply_monitor_transport_aliases:
        canonical_model = MONITOR_MODEL_ALIASES.get(requested_model, canonical_model)
    if reasoning_effort is None and reasoning_tokens is None:
        reasoning_effort = DEFAULT_REASONING_EFFORTS.get(canonical_model.casefold())
    api_provider, provider_model = canonical_model.split("/", 1)
    if api_provider not in {"openai", "google", "openrouter"}:
        raise ValueError(
            f"Unsupported final-protocol monitor provider {api_provider!r}; expected openai, google, or openrouter."
        )

    routing: dict[str, Any] | None = None
    if api_provider == "openrouter":
        normalized_provider_model = provider_model.casefold()
        locked_provider_label = LOCKED_OPENROUTER_UPSTREAM_LABELS.get(normalized_provider_model)
        pinned_upstream = OPENROUTER_UPSTREAMS.get(normalized_provider_model)
        if (
            locked_provider_label is not None
            and openrouter_provider is not None
            and openrouter_provider.casefold() != pinned_upstream
        ):
            raise ValueError(
                f"{provider_model} final runs are pinned to the {locked_provider_label} "
                "OpenRouter provider; "
                "third-party provider overrides are not permitted."
            )
        upstream = openrouter_provider or pinned_upstream
        if not upstream:
            raise ValueError(
                "OpenRouter final runs require an explicit upstream provider pin via --openrouter-provider."
            )
        routing = {
            "only": [upstream],
            "order": [upstream],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    elif openrouter_provider is not None:
        raise ValueError("--openrouter-provider is valid only for openrouter/* models.")
    else:
        upstream = "google-ai-studio" if api_provider == "google" else "openai"

    bare_model = provider_model.casefold()
    is_minimax_m2_5 = (
        api_provider == "openrouter" and bare_model == "minimax/minimax-m2.5"
    )
    if is_minimax_m2_5 and (reasoning_effort is not None or reasoning_tokens is not None):
        raise ValueError(
            "MiniMax M2.5 uses mandatory native reasoning and does not expose "
            "configurable reasoning-effort or reasoning-token settings."
        )
    if (
        reasoning_effort is None
        and reasoning_tokens is None
        and canonical_model.casefold() in ZERO_THINKING_MODELS
    ):
        reasoning_tokens = 0
    is_gemini_three = (
        (api_provider == "google" and bare_model.startswith("gemini-3"))
        or (api_provider == "openrouter" and bare_model.startswith("google/gemini-3"))
    )
    if is_gemini_three:
        if reasoning_tokens is not None:
            raise ValueError(
                "Gemini 3 uses thinking levels; reasoning-token budgets are not reproducible for this model."
            )
        if reasoning_effort in {"xhigh", "max"}:
            raise ValueError("Gemini 3 supports reasoning effort only through high.")

    is_deepseek_v4_flash_0731 = (
        api_provider == "openrouter"
        and bare_model == "deepseek/deepseek-v4-flash-0731"
    )
    if is_deepseek_v4_flash_0731 and reasoning_effort not in {None, "low", "high", "max"}:
        raise ValueError(
            "DeepSeek V4 Flash 0731 supports reasoning effort low, high, or max."
        )

    is_reasoning = reasoning_effort is not None or reasoning_tokens is not None
    omit_sampling = (
        is_gemini_three
        or (provider_model.casefold().startswith("anthropic/claude-sonnet-5"))
        or (provider_model.casefold().startswith("anthropic/claude-haiku-4.5") and is_reasoning)
        or (api_provider == "openai" and bare_model.startswith("gpt-5") and is_reasoning)
    )
    haiku_manual_thinking = (
        api_provider == "openrouter"
        and provider_model.casefold().startswith("anthropic/claude-haiku-4.5")
        and is_reasoning
    )
    qwen_auto_tools = (
        api_provider == "openrouter"
        and provider_model.casefold().startswith("qwen/qwen3.6-flash")
    )
    glm_auto_tools = (
        api_provider == "openrouter"
        and provider_model.casefold().startswith("z-ai/glm-4.5-air")
    )
    minimax_auto_tools = (
        api_provider == "openrouter"
        and provider_model.casefold().startswith("minimax/minimax-m2.5")
    )
    return MonitorModelSpec(
        requested_model=requested_model,
        model=canonical_model,
        api_provider=api_provider,
        upstream_provider=upstream,
        reasoning_effort=reasoning_effort,
        reasoning_tokens=reasoning_tokens,
        requested_temperature=temperature,
        effective_temperature=None if omit_sampling else temperature,
        openrouter_routing=routing,
        tool_choice_policy=(
            "auto_when_tools"
            if haiku_manual_thinking or qwen_auto_tools or glm_auto_tools or minimax_auto_tools
            else "required_when_requested"
        ),
    )


_OPENAI_PATCHED = False
_GOOGLE_PATCHED = False
_OPENROUTER_PATCHED = False


def configure_inspect_reasoning_model(model_name: str) -> None:
    """Install narrow shims for model families newer than Inspect 0.3.93."""

    global _OPENAI_PATCHED, _GOOGLE_PATCHED, _OPENROUTER_PATCHED
    canonical = PINNED_OPENAI_MODELS.get(model_name, model_name)
    provider, bare_name = canonical.split("/", 1)

    if provider == "openai" and bare_name.casefold().startswith("gpt-5") and not _OPENAI_PATCHED:
        from inspect_ai.model._providers import openai as openai_provider
        from inspect_ai.model._providers import openai_responses

        def extend(original: Callable[[str], bool]) -> Callable[[str], bool]:
            def recognizes(name: str) -> bool:
                return name.rsplit("/", 1)[-1].casefold().startswith("gpt-5") or original(name)

            return recognizes

        openai_provider.is_o_series = extend(openai_provider.is_o_series)
        openai_responses.is_o_series = extend(openai_responses.is_o_series)
        _OPENAI_PATCHED = True

    if provider == "google" and bare_name.casefold().startswith("gemini-3") and not _GOOGLE_PATCHED:
        from google.genai.types import ThinkingConfig
        from inspect_ai.model._providers import google as google_provider

        original_thinking_config = google_provider.GoogleGenAIAPI.chat_thinking_config

        def gemini_three_thinking_config(self: Any, config: Any) -> Any:
            effort = getattr(config, "reasoning_effort", None)
            if effort is not None:
                return ThinkingConfig(include_thoughts=True, thinking_level=str(effort).upper())
            return original_thinking_config(self, config)

        google_provider.GoogleGenAIAPI.chat_thinking_config = gemini_three_thinking_config
        _GOOGLE_PATCHED = True

    if provider == "openrouter" and not _OPENROUTER_PATCHED:
        from inspect_ai.model._providers import openrouter as openrouter_provider

        original_completion_params = openrouter_provider.OpenRouterAPI.completion_params

        def openrouter_completion_params(self: Any, config: Any, tools: bool) -> dict[str, Any]:
            params = original_completion_params(self, config, tools)
            reasoning = params.pop("reasoning", None)
            if reasoning is not None:
                # OpenRouter accepts this extension in the JSON body, while
                # openai>=2 rejects it as an unknown Python keyword argument.
                extra_body = dict(params.get("extra_body") or {})
                extra_body["reasoning"] = reasoning
                params["extra_body"] = extra_body
            return params

        openrouter_provider.OpenRouterAPI.completion_params = openrouter_completion_params
        _OPENROUTER_PATCHED = True


_TELEMETRY: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar(
    "monitor_model_telemetry", default=[]
)
_TELEMETRY_ACCUMULATING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "monitor_model_telemetry_accumulating", default=False
)


def reset_monitor_model_telemetry() -> None:
    if not _TELEMETRY_ACCUMULATING.get():
        _TELEMETRY.set([])


def begin_monitor_model_telemetry() -> tuple[contextvars.Token, contextvars.Token]:
    """Start one trajectory scope so retry attempts retain their paid calls."""

    # asyncio.wait_for runs the model coroutine in a child task. Context values
    # are copied into that task, so use one shared list and mutate it in place.
    telemetry_token = _TELEMETRY.set([])
    accumulating_token = _TELEMETRY_ACCUMULATING.set(True)
    return telemetry_token, accumulating_token


def end_monitor_model_telemetry(tokens: tuple[contextvars.Token, contextvars.Token]) -> None:
    telemetry_token, accumulating_token = tokens
    _TELEMETRY_ACCUMULATING.reset(accumulating_token)
    _TELEMETRY.reset(telemetry_token)


def _complete_jsonable(value: Any) -> Any:
    """Serialize a complete provider return without truncating model text."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json", exclude_none=False)
        except TypeError:
            return _complete_jsonable(model_dump())
    if isinstance(value, Mapping):
        return {str(key): _complete_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_complete_jsonable(item) for item in value]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {str(key): _complete_jsonable(item) for key, item in attributes.items()}
    return repr(value)


def _transcript_position() -> tuple[Any | None, int]:
    """Locate Inspect's task-local transcript so raw provider calls can be retained."""

    try:
        from inspect_ai.log._transcript import transcript

        current = transcript()
        return current, len(current.events)
    except Exception:
        return None, 0


def _provider_events_since(current: Any | None, position: int) -> list[Any]:
    if current is None:
        return []
    compact_events: list[Any] = []
    for event in current.events[position:]:
        if getattr(event, "event", None) != "model":
            continue
        serialized = _complete_jsonable(event)
        if isinstance(serialized, dict):
            # The exact prompt is already stored in the monitor transcript and
            # the complete returned ModelOutput is stored in ``response``.
            # Repeating both inside every Inspect provider event can add
            # several megabytes per call without adding audit information.
            serialized.pop("input", None)
            serialized.pop("output", None)
            serialized.pop("tools", None)
            call = serialized.get("call")
            if isinstance(call, dict):
                request = call.get("request")
                if isinstance(request, dict):
                    request.pop("messages", None)
                    request.pop("tools", None)
                response = call.get("response")
                if isinstance(response, dict):
                    # The complete generated choices are already represented
                    # by the call record's ``response``. Retain raw provider,
                    # routing, usage, model, and request-parameter metadata.
                    response.pop("choices", None)
            serialized["input_stored_in_transcript"] = True
            serialized["output_stored_in_response"] = True
        compact_events.append(serialized)
    return compact_events


def monitor_model_telemetry() -> dict[str, Any]:
    calls = list(_TELEMETRY.get())
    token_fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "input_tokens_cache_read",
        "input_tokens_cache_write",
    )
    return {
        "call_count": len(calls),
        "resolved_models": sorted({str(call["resolved_model"]) for call in calls if call.get("resolved_model")}),
        "usage": {
            field: sum(int(call.get(field, 0) or 0) for call in calls)
            for field in token_fields
        },
        "calls": calls,
    }


class _MonitorModelProxy:
    def __init__(self, delegate: Any, spec: MonitorModelSpec) -> None:
        self._delegate = delegate
        self.monitor_model_spec = spec

    def __str__(self) -> str:
        return str(self._delegate)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

        if self.monitor_model_spec.model.casefold() in SYSTEM_ONLY_USER_ROLE_MODELS:
            input_value = args[0] if args else kwargs.get("input")
            if (
                isinstance(input_value, list)
                and len(input_value) == 1
                and isinstance(input_value[0], ChatMessageSystem)
            ):
                system_message = input_value[0]
                user_message = ChatMessageUser(
                    id=system_message.id,
                    content=system_message.content,
                    source=system_message.source,
                    internal=system_message.internal,
                )
                if args:
                    args = ([user_message], *args[1:])
                else:
                    kwargs["input"] = [user_message]

        config = kwargs.get("config")
        if config is None:
            config = GenerateConfig()
        else:
            config = config.model_copy(deep=True)
        if self.monitor_model_spec.effective_temperature is None:
            config.temperature = None
        if self.monitor_model_spec.reasoning_tokens is not None:
            config.reasoning_tokens = self.monitor_model_spec.reasoning_tokens
        if self.monitor_model_spec.reasoning_effort is not None:
            config.reasoning_effort = self.monitor_model_spec.reasoning_effort
        model_max_tokens = MAX_OUTPUT_TOKENS_BY_MODEL.get(
            self.monitor_model_spec.model.casefold()
        )
        if model_max_tokens is not None and (
            config.max_tokens is None or config.max_tokens > model_max_tokens
        ):
            config.max_tokens = model_max_tokens
        kwargs["config"] = config

        if (
            self.monitor_model_spec.tool_choice_policy == "auto_when_tools"
            and kwargs.get("tool_choice") == "any"
        ):
            kwargs["tool_choice"] = "auto"

        call_started = time.perf_counter()
        current_transcript, transcript_position = _transcript_position()
        try:
            output = await self._delegate.generate(*args, **kwargs)
        except Exception as exc:
            record = {
                "call_index": len(_TELEMETRY.get()) + 1,
                "requested_model": self.monitor_model_spec.requested_model,
                "canonical_model": self.monitor_model_spec.model,
                "resolved_model": None,
                "elapsed_seconds": time.perf_counter() - call_started,
                "stop_reason": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "input_tokens_cache_read": 0,
                "input_tokens_cache_write": 0,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "response": None,
                "provider_events": _provider_events_since(current_transcript, transcript_position),
            }
            _TELEMETRY.get().append(record)
            raise
        usage = getattr(output, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        record = {
            "call_index": len(_TELEMETRY.get()) + 1,
            "requested_model": self.monitor_model_spec.requested_model,
            "canonical_model": self.monitor_model_spec.model,
            "resolved_model": getattr(output, "model", None),
            "elapsed_seconds": time.perf_counter() - call_started,
            "stop_reason": getattr(output, "stop_reason", None),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)),
            "reasoning_tokens": int(getattr(usage, "reasoning_tokens", 0) or 0),
            "input_tokens_cache_read": int(getattr(usage, "input_tokens_cache_read", 0) or 0),
            "input_tokens_cache_write": int(getattr(usage, "input_tokens_cache_write", 0) or 0),
            "error": None,
            "response": _complete_jsonable(output),
            "provider_events": _provider_events_since(current_transcript, transcript_position),
        }
        _TELEMETRY.get().append(record)
        return output


def create_monitor_model(spec: MonitorModelSpec) -> Any:
    """Construct an Inspect model with the exact attested provider policy."""

    from inspect_ai.model import GenerateConfig, get_model

    configure_inspect_reasoning_model(spec.model)
    config = GenerateConfig(
        temperature=spec.effective_temperature,
        reasoning_effort=spec.reasoning_effort,
        reasoning_tokens=spec.reasoning_tokens,
    )
    delegate = get_model(spec.model, config=config, **spec.model_args())
    return _MonitorModelProxy(delegate, spec)


def model_spec_metadata(
    spec: MonitorModelSpec,
    *,
    accepted_call_index: int | None = None,
) -> dict[str, Any]:
    telemetry = monitor_model_telemetry()
    calls = [dict(call, accepted_for_score=False) for call in telemetry["calls"]]
    accepted_call = None
    if accepted_call_index is not None:
        accepted_call = next(
            (
                call
                for call in calls
                if call.get("call_index") == accepted_call_index and call.get("response") is not None
            ),
            None,
        )
    elif calls and calls[-1].get("response") is not None:
        accepted_call = calls[-1]
    if accepted_call is not None:
        accepted_call["accepted_for_score"] = True
    return {
        "monitor_requested_model": spec.requested_model,
        "monitor_model": spec.model,
        "monitor_model_api_provider": spec.api_provider,
        "monitor_upstream_provider": spec.upstream_provider,
        "monitor_openrouter_routing": dict(spec.openrouter_routing or {}),
        "monitor_model_spec": spec.as_dict(),
        "monitor_reasoning_effort": spec.reasoning_effort,
        "monitor_reasoning_tokens": spec.reasoning_tokens,
        "monitor_effective_temperature": spec.effective_temperature,
        "monitor_tool_choice_policy": spec.tool_choice_policy,
        "monitor_resolved_models": telemetry["resolved_models"],
        "monitor_model_call_count": telemetry["call_count"],
        "monitor_usage": telemetry["usage"],
        "monitor_model_calls": calls,
        "monitor_accepted_call_index": accepted_call.get("call_index") if accepted_call else None,
        "monitor_accepted_call_usage": (
            {
                field: accepted_call.get(field, 0)
                for field in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "reasoning_tokens",
                    "input_tokens_cache_read",
                    "input_tokens_cache_write",
                )
            }
            if accepted_call
            else None
        ),
    }
