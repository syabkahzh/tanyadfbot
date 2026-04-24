## 2026-04-24 - Performance Optimizations: Python String Concatenation vs. Join

### Learning
In earlier Python versions, string concatenation (`+=`) within a loop could scale quadratically in time because strings are immutable, leading to new objects being created constantly. The recommended approach to mitigate this is storing the items in a list (via comprehension or `append`) and using `"".join(list)`.

### Local Observation
Testing on CPython 3.12 within this repository shows that CPython heavily optimizes local variable string concatenation (`+=`). For small limit datasets (e.g., n=5), string concatenation (`+=`) runs in nearly identical time to `"".join()`. The improvement with `"".join()` only begins to materialize at higher counts (e.g., ~10% faster at n=100 and ~10-20% faster at n=10000).

### Action
We migrated the `cmd_debug` function string builder pattern in `bot.py` from `+=` inside a for-loop to `"".join()` with a list comprehension for safer behavior at scale and to follow standard Pythonic best practices, while documenting the small empirical scaling benefits given the CPython 3.12 environment context.
