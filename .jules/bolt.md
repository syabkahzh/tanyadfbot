## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.
## 2024-05-30 - Optimization of Iterables and Memory Allocation

**Learning:** Allocating a set or list repeatedly inside a high-frequency method (like `_is_worth_checking` which parses thousands of chat messages) incurs a large cumulative memory allocation overhead. Additionally, using Python's generator-based `any()` checks for intersection between lists and sets introduces looping overhead that can be optimized using fast C-level set operations `set_a & set_b`.

**Action:** Elevate locally instantiated sets to the module level. Swap `any(w in TARGET_SET for w in source_list)` with `set(source_list) & TARGET_SET` when both sets are already built or can be cached.
