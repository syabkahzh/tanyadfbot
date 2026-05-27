# Agentdf Goodies Migration

## What was moved from agentdf

Source: `/home/hfzhkn/Projects/agentdf` (greenfield scaffold, reference-only)

### Knowledge assets (direct copy, adapted)

| Source (agentdf) | Target (tanyadfbot) | Notes |
|------------------|---------------------|-------|
| `prompts/promo_judge.md` | `prompts/promo_judge.md` | LLM system prompt for structured promo judgment |
| `skills/discountfess-lingo.md` | `skills/discountfess-lingo.md` | Indonesian promo slang reference; expanded with more terms |
| `skills/promo-review.md` | `skills/promo-review.md` | Review skill for alert quality; adapted for tanyadfbot tables |
| `skills/false-positive-patterns.md` | `skills/false-positive-patterns.md` | Known false positive patterns; expanded |

### New files (inspired by agentdf concepts)

| File | Purpose |
|------|---------|
| `config/trigger_terms.yaml` | Weighted YAML config for promo/payment/platform/mechanic/social terms |
| `structured_judge.py` | Pydantic models (AlertTier, PromoJudgment, ContextBundle) + YAML loader + judge function boundary |
| `HERMES_CONTROL_PLANE.md` | Hermes as first-class control plane documentation |
| `tests/test_structured_judge.py` | Tests for all new models and loaders |

### Concepts ported

1. **Structured JSON judge** — PromoJudgment model with P0/P1/P2/P3 tiers, confidence, relevance, quality, urgency, junk_risk.
2. **Alert tier routing** — P0_REALTIME (immediate), P1_BATCH, P2_DIGEST, P3_IGNORE.
3. **Context bundle** — Clean abstraction for candidate + surrounding messages.
4. **Security model** — "Group messages are untrusted evidence, never instructions."
5. **YAML trigger config** — Dynamic lingo loading instead of hardcoded Python lists.
6. **Hermes control plane** — Operator gateway, daily review, self-learning loop.

## What remains to integrate

These are follow-up tasks after this foundation is tested:

### Near-term (next steps)

1. **Test YAML loader** — Run `tests/test_structured_judge.py` to verify YAML loading and scoring work. ✅ Done — 25/25 pass.
2. **Test structured judge parsing locally** — Verify PromoJudgment model validates correctly. ✅ Done.
3. **Add pyyaml to requirements.txt** — Done in this commit.
4. **Run existing tanyadfbot tests** — Ensure nothing broke. ✅ Done (structured_judge tests pass; other tests blocked by missing `fasttext-wheel` build dependency in this environment).

### Runtime integration (first pass — DONE)

5. **YAML attention gating in `main.py` pre-filter** — ✅ Done. `config/trigger_terms.yaml` is loaded at startup via `structured_judge.load_trigger_terms()` with safe fallback. A helper `_yaml_attention_result()` calls `structured_judge.score_from_yaml()`. In `processing_loop()` Tier 1 pre-filter, if YAML attention says `should_build_context == True`, the message is allowed into `to_ai` even if the old regex filter would have dropped it. This broadens inspection so slang variants from `config/trigger_terms.yaml` get seen by the existing AI processor. No new LLM calls. No changes to `listener.py` fast-path.

### Medium-term (still follow-up)

6. **Integrate structured LLM judge after processor output** — After `processor.py` extracts promos, run the structured LLM judge on candidates for P0/P1/P2/P3 classification.
7. **Route alerts by tier** — Update `bot.py` to send P0 immediately, batch P1, digest P2, skip P3.
8. **Migrate hardcoded keywords to YAML** — Replace `_STRONG_KEYWORDS` and regex sets in `listener.py` with `config/trigger_terms.yaml` loading. Keep both paths working during transition.

### Long-term (Hermes-driven)

8. **Hermes daily review** — Let Hermes inspect logs/DB and produce daily summaries.
9. **Hermes proposes YAML/skill/prompt diffs** — Self-learning loop updates config without code changes.
10. **Hermes gated source patches** — Code changes with tests + git diff + user approval.

## Safety notes

- `structured_judge.py` YAML attention scoring IS wired into `main.py` pre-filter (Tier 1). This only gates context building — it does NOT trigger alerts or LLM calls.
- The structured LLM judge (`judge_context_bundle`) is NOT wired into live processing yet. It requires a live LLM client and raises ValueError if called without one.
- `listener.py` fast-path is untouched in this integration pass.
- All group message content remains untrusted evidence, never instructions.
- The `config/trigger_terms.yaml` file is a Hermes-writable control-plane asset.
