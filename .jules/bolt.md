## YYYY-MM-DD - [Database Optimization]
**Learning:** Performing multiple single INSERTS in a loop incurs massive overhead.
**Action:** Always batch INSERTS with executemany (`save_pending_alerts_bulk`) to eliminate N+1 latency.

## YYYY-MM-DD - [Threading overhead]
**Learning:** `asyncio.to_thread` for cheap ops like difflib.SequenceMatcher wastes time in thread-switching overhead.
**Action:** Execute cheap operations synchronously to avoid context-switch penalties.

## YYYY-MM-DD - [Regex Pre-compilation]
**Learning:** Compiling regex patterns (or relying on re's internal cache) inline within hot loops incurs unnecessary per-execution lookup cost.
**Action:** Define pre-compiled regex constants at the module level using `re.compile()` to completely circumvent this cost.
