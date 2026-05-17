## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.
## 2025-02-12 - Re.compile in Processor and Jobs

**Learning:** `re.sub` and `re.search` calls in frequently accessed processor logic, listener events, and asynchronous jobs cause unnecessary overhead by repeatedly checking the internal Python regex compilation cache instead of directly invoking compiled matchers. Inline patterns can be 1.8x+ slower to evaluate than `re.compile()` constants.

**Action:** In addition to search operations (`re.search`), ensure all `re.sub` patterns (e.g. for tag cleanup or formatting) and complex condition checks are hoisted as `re.compile()` variables at the module level.
