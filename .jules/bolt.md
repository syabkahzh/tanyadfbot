## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.
## 2024-05-30 - Fix N+1 query via `executemany` bulk insert

**Learning:** When generating a high volume of alerts within asynchronous job loops (like `confirmation_gate_job`), row-by-row `await db.save_pending_alert` triggers the "N+1 query problem", which creates extensive blocking and overhead for SQLite, even with WAL enabled.

**Action:** Accumulate row data in a list `alerts_to_save: list[dict[str, Any]]` within the loop, then call a single batched method `db.save_pending_alerts_bulk(alerts_to_save)` that utilizes `await self.conn.executemany` after the loop. This minimizes overhead entirely.
