# TanyaDFBot

TanyaDFBot is a high-performance Telegram bot designed for real-time promotion tracking, sentiment analysis, and community management. It leverages Gemini AI to extract deals, identify trends, and provide community-driven corroboration for promotions.

## Core Features
- **Instant Deal Detection**: Uses pattern-based "Fast-Path" matching for immediate notification of high-signal keywords (e.g., cashback, vouchers).
- **AI-Powered Analysis**: Utilizes Google Gemini models to parse raw chat text, extract structured promotion data, and summarize discussions.
- **Community Corroboration**: Features a "Confirmation Gate" that waits for multiple user reports before broadcasting a deal, reducing false positives.
- **Thread Analysis**: Automatically identifies and summarizes "hot" discussion threads.
- **Feedback Pipeline**: Allows owners to provide direct corrections via a **"🔧 Feedback"** button, creating a ground-truth dataset for AI learning. See [Feedback Skill](.agents/skills/feedback-pipeline/SKILL.md).
- **Resilient Pipeline**: Built with robust asynchronous message ingestion, batch database processing, and auto-reconnection logic.

## Architecture
TanyaDFBot is built as a monolithic `asyncio` application:
- **Listener Layer**: Uses `Telethon` for real-time Telegram event ingestion.
- **Processing Layer**: Employs `GeminiProcessor` to perform intelligent, batch-based message analysis.
- **Bot Layer**: Handles Telegram interactions, alert formatting, and command dispatching.
- **Database Layer**: Uses `aiosqlite` with WAL mode and intelligent triage to ensure high-speed, thread-safe data persistence.

## Key Performance Optimizations
- **Non-blocking Pipeline**: All I/O-heavy operations (file access, sub-processes) are offloaded to `asyncio.to_thread`.
- **Batch Commits**: Database writes are buffered and committed in intervals to minimize disk I/O contention.
- **AI Throttling**: Intelligent concurrency limiting (semaphores) prevents AI API saturation and event loop stalls.
- **Intelligent Deduplication**: Uses memory-locked history to ensure no duplicate alerts are broadcasted for the same deal.

## Documentation
Detailed project knowledge and AI-generated insights are maintained in the **[Devin DeepWiki](https://app.devin.ai/org/hafizh-survey-2ee2579f8b07463cbc32e7587f8384f4/wiki/syabkahzh/tanyadfbot?branch=master)**.

## Project Structure
- `main.py`: Entry point and orchestrator.
- `bot.py`: Telegram command handlers and broadcaster.
- `listener.py`: Telethon-based message listener and pipeline ingress.
- `processor.py`: AI integration and NLP logic.
- `jobs.py`: Background tasks (digests, trend monitoring, spike detection).
- `db.py`: Database interaction, schema management, and recovery.
- `shared.py`: Global state and cross-module utilities.

## Tools

### `tools/export_corrections.py`
Exports the latest AI corrections (user feedback) from the database into a readable Markdown report (`latest_feedback_export.md`).

**Usage:**
```bash
python tools/export_corrections.py
```

**Automation (VPS):**
A cron job runs this every 6 hours on the VPS so `latest_feedback_export.md` is always up-to-date:
```bash
0 */6 * * * cd /home/hfzhkn/tanyadfbot && /home/hfzhkn/tanyadfbot/venv/bin/python tools/export_corrections.py >> /home/hfzhkn/tanyadfbot/logs/export_cron.log 2>&1
```

## Deployment
The project is managed via a Git repository. Deployment is handled through a custom `deploy.sh` script that performs incremental syncs and service restarts.

---
*Maintained with ✨ by TanyaDFBot Team*
