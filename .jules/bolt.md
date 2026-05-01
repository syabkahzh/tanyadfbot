## 2024-05-24 - Pre-compiled Regexes for Substring Matching
**Learning:** For testing multiple literal substring matches inside a frequently called loop (e.g., `_is_worth_checking` running on every message), a single pre-compiled regex utilizing alternation (`re.compile('|'.join(map(re.escape, WORDS)))`) is approximately 45-60% faster in Python than evaluating a generator expression like `any(kw in text for kw in WORDS)`.

**Action:** When evaluating sets of keyword substrings on hot paths, transform the sets into a single compiled regex and use `pattern.search()`. Always ensure to use `re.escape()` to correctly preserve literal matching semantics. Also, module-level sets used for membership testing should be declared as `frozenset` at the module level rather than being dynamically created inline.

## 2024-05-30 - Optimization of RegEx compilation in Job
**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.
**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.

## 2023-10-27 - [Precompile Regex in Batch Processing Loops]
**Learning:** `re.findall` and `re.compile` calls inside tight nested loops across history caches cause high processing latency and create O(N*M) redundant executions. Specifically, applying `re.findall(r'\w+', ...)` to the same history tail element repeatedly for every item in a batch creates an enormous overhead.
**Action:** Always precompile heavily used regex expressions globally (`_WORDS_PATTERN = re.compile(r'\w+')`), and precompute / cache their results before entering inner loops involving cross-comparing data (e.g., $O(N \times M)$ comparisons in deduplication).

## 2025-05-18 - [Optimization & Stability: Regex & Exceptions]
**Learning:** Pre-compiling commonly used regular expressions improves application throughput. Additionally, using bare `except:` clauses is dangerous as it can swallow `KeyboardInterrupt` and `SystemExit`.
**Action:** Always declare heavy regular expressions at module scope using `re.compile()`, and explicitly catch specific exceptions (like `ValueError` or `json.JSONDecodeError`) instead of relying on a catch-all `except:`.
