## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.
## 2024-05-18 - Logic Preservation during Optimization
**Learning:** When refactoring nested loops to avoid O(N*M) bottlenecks (like grouping history arrays into dicts in `processor.py`), it is critical to keep the gating logic separate. Grouping `p.status == 'active'` and `brand_key != 'unknown'` caused a regression where non-active promos incorrectly skipped the intra-batch duplicate check.
**Action:** When extracting checks out of a nested loop, carefully verify that `if` conditions are not overly merged. Keep independent state validations (like 'status') decoupled from group-key checks (like 'brand_key') so unrelated branches are not bypassed.
