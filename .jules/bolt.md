## Memory Update
- `check_fast_path` and `_is_worth_checking` were substantially improved to correctly identify questions (e.g. `aman ga ya`) versus strong signals (`aman spx`).
- Replaced boolean-heavy `_is_worth_checking` with a fast heuristic scoring approach (+ points for strong/promo keywords, - points for questions/noise).
- Both logic sets now enforce filtering on social expressions (like `makasih kak` or `ya allah😭`) lacking actual promotional signals.
## 2024-05-30 - Optimization of RegEx compilation in Job

**Learning:** Pre-compiling RegEx locally within a frequently called asynchronous job function forces the Python interpreter to look up or recompile the regular expression repeatedly, causing unnecessary overhead. While Python caches RegEx compilations, relying on module-level constants circumvents the lookup entirely.

**Action:** Consistently elevate pre-compiled `re.compile` patterns to the module level instead of defining them locally within functions, especially inside asynchronous loops or frequent jobs.
## 2025-05-18 - [Optimization & Stability: Regex & Exceptions]
**Learning:** Pre-compiling commonly used regular expressions improves application throughput. Additionally, using bare `except:` clauses is dangerous as it can swallow `KeyboardInterrupt` and `SystemExit`.
**Action:** Always declare heavy regular expressions at module scope using `re.compile()`, and explicitly catch specific exceptions (like `ValueError` or `json.JSONDecodeError`) instead of relying on a catch-all `except:`.
