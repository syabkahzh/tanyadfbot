# TanyaDFBot Project Handover

## Project Overview
TanyaDFBot is a high-performance, AI-powered Telegram deal aggregator. It monitors a target group chat, extracts promotions using the Gemini API (Gemma model), and sends structured alerts to the owner.

## System Architecture
The system follows an **Ingest -> Store -> Extract -> Alert** pipeline.

### 1. Ingestion (`listener.py`)
- **Engine:** Telethon (Telegram Client).
- **Mechanism:** Listens for `NewMessage` events and performs a non-blocking background history sync (up to 12 hours) on startup.
- **Storage:** Messages are saved to the `messages` table with `processed = 0`.

### 2. Storage (`db.py`)
- **Engine:** `aiosqlite` with a **single persistent connection**.
- **Mode:** `PRAGMA journal_mode=WAL` (Write-Ahead Logging) to allow concurrent read/write.
- **Atomicity:** Uses `save_promos_batch` for transactional commits (saves promos and marks messages as processed in one transaction).

### 3. AI Processing (`processor.py` & `main.py`)
- **Primary Model:** `gemma-3-4b-it` (Gemma 3).
- **Fleet:** AI Army (multiple providers including Gemma 3 [4b, 12b, 27b], Gemma 4, Mistral, Llama, Qwen).
- **Extraction:** Processes messages in batches of 150 every 1 minute.
- **Deduplication:** Uses **Batch AI Deduplication**. A single AI call compares a batch of new promos against recent history to filter out duplicates semantically.
- **Alert Logic:**
    - **Freshness:** 60-minute cutoff for alerts.
    - **Catch-up:** During the first 5 minutes of startup (`is_booting`), the 60-minute rule is ignored to notify of deals found during history sync.

### 4. Interaction (`bot.py`)
- **Engine:** `python-telegram-bot`.
- **UI:** Structured alerts with WIB timestamps, "Mark Expired" interactive buttons, and `/summary` lists.
- **Commands:**
    - `/aisummary [n]`: AI-centric raw text summary of the last $N$ messages.
    - `/summary [jam]`: List of extracted promos from the last $X$ hours.
    - `/status`: Real-time system health and queue depth.
    - `/today`: All promos from the last 24 hours.

## Technical Standards
- **Timezone:** Internal logic and database use **UTC**. Presentation layer (`bot.py`) adds a **+7 hour offset** for **WIB**.
- **Data Integrity:** No messages are marked as processed until the AI successfully extracts promos and they are saved to the DB.
- **Deployment:** Uses `rsync` with exclusions for `.db`, `.session`, and `.env`.

## Current State
- **Performance:** Optimized for high-volume groups with batch processing and persistent DB connections.
- **Robustness:** Handles service restarts and history catch-ups without losing data or silencing important deals.
- **Slang Sensitivity:** AI instructions are tuned for Indonesian "deal hunter" slang (e.g., "pln on", "aman", "nyantol").

Ran on GCP's VPS
