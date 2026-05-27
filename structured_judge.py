"""Structured promo judgment models and judge function boundary.

Ported from agentdf's models.py and judge.py concepts.
This module is isolated — not wired into production main.py yet.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# --- Alert tiers ---

class AlertTier(str, Enum):
    P0_REALTIME = "P0_REALTIME"
    P1_BATCH = "P1_BATCH"
    P2_DIGEST = "P2_DIGEST"
    P3_IGNORE = "P3_IGNORE"


# --- Context bundle ---

class ContextBundle(BaseModel):
    """Context around a candidate message for LLM judgment."""
    chat_id: str
    candidate_msg_id: int
    candidate: dict[str, Any]
    replied_to: dict[str, Any] | None = None
    before: list[dict[str, Any]] = Field(default_factory=list)
    after: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --- Structured judgment output ---

class PromoJudgment(BaseModel):
    """Structured output from the LLM promo judge."""
    is_promo: bool
    confidence: float = Field(ge=0, le=1)
    promo_relevance: float = Field(ge=0, le=1)
    deal_quality: float = Field(ge=0, le=1)
    urgency: float = Field(ge=0, le=1)
    junk_risk: float = Field(ge=0, le=1)
    alert_tier: AlertTier

    deal_type: list[str] = Field(default_factory=list)
    merchant: str | None = None
    platform: str | None = None
    payment_method: str | None = None
    product_or_category: str | None = None

    minimum_spend: str | None = None
    maximum_discount: str | None = None
    discount_rate: str | None = None
    voucher_code: str | None = None
    expiry: str | None = None
    is_targeted: bool | None = None

    evidence_message_ids: list[int] = Field(default_factory=list)
    reasoning_summary: str
    caveats: list[str] = Field(default_factory=list)
    junk_reason: str | None = None


# --- YAML trigger loader ---

DEFAULT_TRIGGER_PATH = Path(__file__).parent / "config" / "trigger_terms.yaml"


def load_trigger_terms(path: Path | str = DEFAULT_TRIGGER_PATH) -> dict[str, Any]:
    """Load trigger terms from YAML config.

    Returns a dict with category names as keys, each containing
    'weight' (float) and 'terms' (list[str]).

    Example::

        terms = load_trigger_terms()
        for cat, data in terms.items():
            print(cat, data["weight"], len(data["terms"]))
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Validate structure
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict at top level, got {type(raw).__name__}")

    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Category '{name}': expected dict, got {type(entry).__name__}")
        if "weight" not in entry:
            raise ValueError(f"Category '{name}': missing 'weight'")
        if "terms" not in entry:
            raise ValueError(f"Category '{name}': missing 'terms'")
        if not isinstance(entry["terms"], list):
            raise ValueError(f"Category '{name}': 'terms' must be a list")

    return raw


def score_from_yaml(text: str, trigger_terms: dict[str, Any], has_media: bool = False) -> dict[str, Any]:
    """Score text using YAML-loaded trigger terms.

    Returns dict with 'score' (float), 'reasons' (list[str]), 'should_build_context' (bool).
    """
    t = (text or "").lower()
    reasons: list[str] = []
    score = 0.0

    for name, data in trigger_terms.items():
        hits = [term for term in data["terms"] if term in t]
        if hits:
            score += data["weight"]
            reasons.append(f"{name}:{','.join(hits[:5])}")

    if has_media:
        score += 1.5
        reasons.append("media")

    return {
        "score": score,
        "reasons": reasons,
        "should_build_context": score >= 2.0,
    }


# --- Judge function boundary ---

JUDGE_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "promo_judge.md"


def load_judge_system_prompt(path: Path | str = JUDGE_SYSTEM_PROMPT_PATH) -> str:
    """Load the judge system prompt from markdown file."""
    with open(path) as f:
        return f.read()


async def judge_context_bundle(
    bundle: ContextBundle,
    *,
    system_prompt: str | None = None,
    client: Any = None,  # AsyncOpenAI instance
    model: str = "mimo-v2.5",
) -> PromoJudgment:
    """Judge a context bundle using an OpenAI-compatible LLM.

    This is a function boundary — it wraps the LLM call and validates
    the response into a PromoJudgment. Does NOT make API calls in tests
    unless a real client is passed.

    Args:
        bundle: Context bundle with candidate + surrounding messages.
        system_prompt: Override system prompt. Loads from file if None.
        client: AsyncOpenAI instance. If None, raises ValueError.
        model: Model ID to use.

    Returns:
        Validated PromoJudgment.

    Raises:
        ValueError: If no client provided.
        Exception: On LLM or validation errors.
    """
    if client is None:
        raise ValueError("client (AsyncOpenAI) is required for live judgment")

    if system_prompt is None:
        system_prompt = load_judge_system_prompt()

    import json

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(bundle.model_dump(), default=str)},
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    return PromoJudgment(**data)
