## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.

## 2025-02-20 - [Pre-compute historical tokenization in duplicate filter]
**Learning:** O(N*M) performance bottlenecks can easily happen when filtering batch items (`new_promos`) against a historical collection (`history_tail`) if string operations or regular expressions are executed inside the inner loop.
**Action:** Always pre-compute repeated operations on static collections before entering the iteration loop, grouping results by a shared lookup key to enable fast O(1) evaluation.
