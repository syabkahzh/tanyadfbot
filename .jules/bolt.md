## 2024-05-14 - SQLite Database Performance Optimization
**Learning:** Adding an index on frequently queried columns in SQLite (like `pending_confirmations(brand)`) can yield significant performance improvements, turning an O(n) scan into an O(1) or O(log n) lookup. However, running any scripts that connect to SQLite may generate binary database files (e.g. `.db-shm`, `.db-wal`) that should NOT be committed to the repository to avoid cluttering version control.
**Action:** Always check `git status` after running local scripts and explicitly unstage/remove any generated SQLite auxiliary binary files before creating a commit.
## 2026-04-24 SQLite Batch Update Optimization
* **Context**: `image_processing_job` iterating through a maximum 5-item batch sequence of pending image tasks.
* **Anti-Pattern**: Using loop-driven sequential queries in `.execute` paired with immediate `.commit()` statements for tracking state (e.g. `image_processed=1`) creates excessive transactional locking overhead (N+1 queries).
* **Optimization**: Collect `msg_id` variables into a `processed_ids` list during iteration. Perform a single bulk update via `UPDATE ... WHERE id IN (...)` wrapped in a single transaction at the tail of the function loop.
* **Outcome**: Yields roughly an 80% decrease in transactional context switching with SQLite per execution tick (from 5 `.commit` statements maximum down to 1), heavily optimizing CPU usage and potential write locks.
