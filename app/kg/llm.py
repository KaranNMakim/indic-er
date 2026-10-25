"""LLMClient: an interface, a deterministic mock, and a live Anthropic client.

Design rule for this project, stated in PLAN.md and enforced by the import
graph: **the matching core never calls an LLM.** ``app/core`` does not import
this module. Matching must be reproducible, offline, cheap and defensible to
an auditor — three things an LLM call is not.

The LLM is used for exactly two optional jobs:

1. ``explain_pair``  — rephrase the rule-based explanation more fluently.
2. ``adjudicate_pair`` — offer a second opinion on a borderline pair, as
   advice for the human reviewer. It never moves a score on its own.

``LLM_MODE=mock`` (the default) is deterministic: the same input always yields
the same output, so tests and the demo need no API key and no network.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an entity-resolution assistant for Indian names, businesses and "
    "addresses. You explain why two records may or may not be the same "
    "real-world entity. Be concise and concrete, name the evidence, and never "
    "invent a field that was not given to you."
)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    mode: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Adjudication:
    """A second opinion on a borderline pair. Advice, never a decision."""

    verdict: str  # "same" | "different" | "unsure"
    confidence: float
    rationale: str
    model: str
    mode: str
    advisory: bool = field(default=True)


class LLMClient(ABC):
    """Everything the project is allowed to ask of an LLM."""

    mode: str = "mock"
    model: str = "none"

    @abstractmethod
    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> LLMResponse: ...

    def explain_pair(self, payload: dict[str, Any]) -> LLMResponse:
        """Turn a feature breakdown into a sentence a human would write."""
        return self.complete(
            "Two records may describe the same entity. Here is the feature "
            "breakdown produced by a rule-based matcher:\n\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
            "In at most three sentences, explain to a non-technical reviewer "
            "why these records were or were not matched. Name the strongest "
            "and weakest pieces of evidence.",
            system=SYSTEM_PROMPT,
        )

    def adjudicate_pair(self, payload: dict[str, Any]) -> Adjudication:
        """Second opinion on a borderline pair."""
        response = self.complete(
            "A rule-based matcher could not decide whether these two records "
            "are the same entity:\n\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
            'Reply with JSON only: {"verdict": "same|different|unsure", '
            '"confidence": 0.0-1.0, "rationale": "one sentence"}.',
            system=SYSTEM_PROMPT,
            max_tokens=300,
        )
        return _parse_adjudication(response)


def _parse_adjudication(response: LLMResponse) -> Adjudication:
    text = response.text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            verdict = str(data.get("verdict", "unsure")).lower()
            if verdict not in {"same", "different", "unsure"}:
                verdict = "unsure"
            return Adjudication(
                verdict=verdict,
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                rationale=str(data.get("rationale", "")).strip(),
                model=response.model,
                mode=response.mode,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning("llm adjudication was not parseable JSON; returning 'unsure'")
    return Adjudication(
        verdict="unsure",
        confidence=0.5,
        rationale=text[:300],
        model=response.model,
        mode=response.mode,
    )


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #


class MockLLMClient(LLMClient):
    """Deterministic, offline, no API key.

    ``explain_pair`` and ``adjudicate_pair`` are overridden to build their
    answer from the payload itself, so the demo reads sensibly rather than
    showing placeholder text — and it reads the same way every run.
    """

    mode = "mock"
    model = "mock-deterministic"

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> LLMResponse:
        digest = hashlib.sha256(f"{system or ''}\n{prompt}".encode()).hexdigest()[:12]
        return LLMResponse(
            text=(
                f"[mock-llm:{digest}] Deterministic stand-in for a live completion. "
                "Set LLM_MODE=live and ANTHROPIC_API_KEY to use the real model."
            ),
            model=self.model,
            mode=self.mode,
            input_tokens=len(prompt) // 4,
            output_tokens=24,
        )

    def explain_pair(self, payload: dict[str, Any]) -> LLMResponse:
        features: dict[str, Any] = payload.get("features") or {}
        scored = [
            (name, info)
            for name, info in features.items()
            if isinstance(info, dict) and info.get("value") is not None
        ]
        scored.sort(key=lambda kv: float(kv[1].get("value", 0.0)), reverse=True)
        score = float(payload.get("score") or 0.0)
        decision = str(payload.get("decision", "review"))

        verdict = {
            "merge": "These two records look like the same entity",
            "review": "These two records are close but not conclusive",
            "reject": "These two records look like different entities",
        }.get(decision, "These two records were compared")

        if scored:
            top = ", ".join(
                f"{_pretty(name)} ({float(info['value']):.0%})" for name, info in scored[:2]
            )
            weakest_name, weakest_info = scored[-1]
            weakest = f"{_pretty(weakest_name)} ({float(weakest_info['value']):.0%})"
            body = (
                f" The strongest evidence is {top}. The weakest is {weakest}."
                if len(scored) > 1
                else f" The only usable evidence is {top}."
            )
        else:
            body = " No feature could be computed — the records share no comparable field."

        text = (
            f"{verdict} (score {score:.2f}, decision '{decision}')."
            f"{body} Matching used rules only; no model was consulted."
        )
        return LLMResponse(text=text, model=self.model, mode=self.mode, output_tokens=40)

    def adjudicate_pair(self, payload: dict[str, Any]) -> Adjudication:
        score = float(payload.get("score") or 0.0)
        if score >= 0.80:
            verdict, confidence = "same", 0.8
        elif score <= 0.45:
            verdict, confidence = "different", 0.75
        else:
            verdict, confidence = "unsure", 0.5
        return Adjudication(
            verdict=verdict,
            confidence=confidence,
            rationale=(
                f"Deterministic mock adjudication derived from the rule score "
                f"({score:.2f}). Set LLM_MODE=live for a real second opinion."
            ),
            model=self.model,
            mode=self.mode,
        )


def _pretty(feature_name: str) -> str:
    return feature_name.replace("_", " ")


# --------------------------------------------------------------------------- #
# Live (Anthropic)
# --------------------------------------------------------------------------- #


class AnthropicLLMClient(LLMClient):
    """Live client. Requires ``anthropic`` installed and an API key."""

    mode = "live"

    def __init__(self, model: str, api_key: str | None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LLM_MODE=live needs the anthropic SDK: pip install anthropic"
            ) from exc
        if not api_key:
            raise RuntimeError("LLM_MODE=live needs ANTHROPIC_API_KEY to be set")
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> LLMResponse:  # pragma: no cover - needs network and a key
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(
            text=text.strip(),
            model=self.model,
            mode=self.mode,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


def build_llm_client(config: Settings | None = None) -> LLMClient:
    """Factory driven by ``LLM_MODE``. Falls back to the mock on any problem."""
    config = config or settings()
    if config.llm_mode == "live":
        try:
            client = AnthropicLLMClient(config.anthropic_model, config.anthropic_api_key)
            log.info("llm_client_ready mode=live model=%s", client.model)
            return client
        except RuntimeError as exc:
            log.warning("llm live mode unavailable (%s) — falling back to mock", exc)
    log.info("llm_client_ready mode=mock")
    return MockLLMClient()
