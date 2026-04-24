## 2026-04-25 Performance Optimization - main.py N+1 Query in pending_confirmations

**What:**
- Modified the loop processing `low_confidence_promos` to perform batch lookups using `WHERE brand IN (...)`.
- Grouped `INSERT` and `UPDATE` queries using `executemany` instead of single queries inside the loop.

**Why:**
- Doing a DB lookup (`SELECT ... LIMIT 1`), then an `UPDATE` or `INSERT` per message in the loop causes an N+1 query problem, increasing lock contention and reducing DB throughput during bursts.

**Measured Improvement:**
- `process_one_batch` now only issues exactly 1 `SELECT`, 1 `UPDATE`, and 1 `INSERT` execution for `pending_confirmations`, regardless of batch size instead of up to N of each.
