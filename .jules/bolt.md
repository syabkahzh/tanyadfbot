## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.
## 2024-05-12 - N+1 Query Fix in Background Jobs
**Learning:** Found an N+1 database insertion pattern in the `confirmation_gate_job` in `jobs.py`. It iterated through a list and called an async `db.save_pending_alert` method on every iteration. While async overhead is somewhat mitigated, doing many discrete inserts incurs massive per-transaction roundtrip overhead for `aiosqlite`.
**Action:** Always inspect loops containing `await db...` inserts/updates. If multiple items belong to the same logical operation, aggregate them into a list in Python memory and create a matching `_bulk` method in `db.py` to use `executemany` with a single transaction.
