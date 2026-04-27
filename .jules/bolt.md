## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.

## 2023-10-27 - [Precompile Regex in Batch Processing Loops]
**Learning:** `re.findall` and `re.compile` calls inside tight nested loops across history caches cause high processing latency and create O(N*M) redundant executions. Specifically, applying `re.findall(r'\w+', ...)` to the same history tail element repeatedly for every item in a batch creates an enormous overhead.
**Action:** Always precompile heavily used regex expressions globally (`_WORDS_PATTERN = re.compile(r'\w+')`), and precompute / cache their results before entering inner loops involving cross-comparing data (e.g., $O(N \times M)$ comparisons in deduplication).
