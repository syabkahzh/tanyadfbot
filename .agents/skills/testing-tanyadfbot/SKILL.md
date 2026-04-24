# Testing TanyaDFBot Pipeline

How to test TanyaDFBot's message processing pipeline (regex filters, temporal context, dedup, LLM prompts).

## Architecture Overview

The bot has 3 message processing paths:
1. **Fast-path** (`listener.py`): `INSTANT_PATTERN` + `NEG_PATTERN` + transit-noise gate → instant alerts without AI
2. **Pre-filter** (`processor.py`): `_STRONG_KEYWORDS` + `_is_worth_checking()` → decides if message goes to AI
3. **AI-path** (`processor.py` → `main.py`): Gemini LLM extraction → DB write → broadcast

Key supporting modules:
- `shared.py`: `TemporalBrandTracker` (TTL cache), `TRANSIT_NOISE_PATTERN`, `is_fuzzy_duplicate()`
- `jobs.py`: Scheduled tasks including `time_mention_job()` with `_is_time_signal_worthy()`
- `db.py`: Brand canon mappings (`_BRAND_CANON`)

## Testing Strategy

All testing is **shell-based** (Python scripts). No UI exists — the bot runs as a Telegram listener. No screen recording needed.

### Import Isolation

**Critical**: You cannot `import processor`, `import jobs`, or `import shared` directly because they transitively import `GeminiProcessor` which requires `GEMINI_API_KEY` at module load time.

**Workaround**: Replicate the specific functions/patterns you need to test in your test script. Copy the exact regex patterns and function logic from the source files. This is reliable because:
- Regex patterns are self-contained
- Functions like `_is_time_signal_worthy()` only depend on regex patterns
- `TemporalBrandTracker` and `is_fuzzy_duplicate()` are pure logic with no external deps

Alternatively, you can extract patterns by reading the source file and using `re.search()` to find pattern strings, but be careful with quote escaping in multi-line patterns.

### What to Test

1. **Sinyal Waktu filter** (`_is_time_signal_worthy` in `jobs.py`): Tests 5-layer gating — questions, speculation, retrospective, complaints, brand+signal requirement. Use the user's reported false positives as test vectors.

2. **Fast-path regex** (`INSTANT_PATTERN`, `NEG_PATTERN` in `listener.py`, `TRANSIT_NOISE_PATTERN` in `shared.py`): Test new activation/negation signals and the transit-noise gate for "aman".

3. **Temporal Brand Context** (`TemporalBrandTracker` in `shared.py`): Test TTL expiry (use short TTL like 2s for testing), eviction at max_chats cap, "Unknown" not overwriting valid brands.

4. **Fuzzy Dedup** (`is_fuzzy_duplicate` in `shared.py`): Test similarity threshold (60%), cross-brand behavior, window expiry. Note: `difflib.SequenceMatcher` similarity can be unintuitive — always compute the actual similarity before writing expected values.

5. **Keyword presence** (`_STRONG_KEYWORDS` in `processor.py`): Verify new slang terms are present. Note: activation STATUS signals (`gacor`, `mantul`, `nyala`, `cair`, `lancar`) belong in `INSTANT_PATTERN`, not `_STRONG_KEYWORDS`.

6. **LLM prompt content** (`_EXTRACT_SYSTEM` in `processor.py`): Verify prompt text contains required rules and examples.

## Indonesian Language Gotchas

- **`-nya` suffix**: Indonesian possessive (e.g., "rutenya" = "the route"). Regex `\b` word boundaries won't match "rutenya" if the pattern only has "rute". Use optional suffix groups: `(rute|jalan|...)(nya|an)?`
- **`-an` suffix**: Indonesian noun/adjective form (e.g., "jalanan" = "streets"). Same fix as `-nya`.
- **Slang contractions**: `gak`/`nggak`/`ngga` = "no", `kyny`/`kynya` = "kayaknya" (maybe), `kmrn`/`kmren` = "kemarin" (yesterday)

## Devin Secrets Needed

- `GEMINI_API_KEY` — Only needed for full end-to-end AI path testing. Not needed for regex/filter/dedup testing.
- `TELEGRAM_BOT_TOKEN` — Only needed for live bot testing. Not needed for logic testing.
- `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` — Only needed for Telethon listener testing.

## Tips

- When testing `is_fuzzy_duplicate`, always compute `difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()` first to set correct expectations. Similarity values can be surprising.
- The `TemporalBrandTracker` is async — use `asyncio.run()` in test scripts.
- For `_is_time_signal_worthy`, the function requires BOTH a brand match AND a signal/status match to return True. This is intentionally strict.
- `_STRONG_KEYWORDS` is extracted as a Python set literal — you can `eval()` the brace-delimited block from the source file.
