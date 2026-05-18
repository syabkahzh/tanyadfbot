## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.
## 2025-03-02 - O(N*M) Regex Bottleneck in Duplicate Filtering
**Learning:** `processor.py` processes duplicates by doing inner-loop regular expressions to extract words from historical alerts `re.findall(r'\w+', r['summary'].lower())[:8]` resulting in O(N*M) redundant computations and cache thrashing. This is specific to how batching operates on historical tails and was a notable slowdown.
**Action:** When performing cross-comparisons between an active batch and a historical tail, always pre-compute transformations (like regex word tokenization) over the history array and map them by key (e.g., brand) before the nested loop. This converts O(N*M) to O(N+M) and removes the compiler overhead from the critical path.
