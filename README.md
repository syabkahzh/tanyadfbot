# 🤖 TanyaDFBot: AI-Powered Deal Hunter & Group Analyst

TanyaDFBot is a high-performance, resilient Telegram deal aggregator designed for the **Discountfess (DF)** ecosystem. It monitors high-volume group chats, extracts promotions using **Gemma/Gemini AI**, and provides real-time alerts, automated digests, and intelligent conversation analysis.

## 🚀 Key Features

### 🧠 Intelligent Extraction
*   **Multimodal Vision Pipeline:** Automatically "sees" and analyzes promotional posters and screenshots using Gemma Vision models.
*   **Slang Sensitivity:** Tuned for Indonesian "deal hunter" slang (e.g., *jp, mm, bs, aman, nt*).
*   **Contextual Awareness:** Uses bulk context queries to understand brief messages (like "on" or "aman") by reading surrounding conversation.

### 🛡️ Unrivaled Reliability
*   **Bulletproof Deduplication:** A 3-layer guard (RAM History + Prefix Matching + DB Unique Index) ensures you never get the same alert twice.
*   **Persistent Alert Buffer:** Alerts are stored in the database before sending, ensuring zero notification loss even if the server crashes.
*   **Smart Re-sync:** Detects downtime and automatically catches up on the last 90 minutes of relevant history without flooding you with stale data.

### 📊 Automated Insights
*   **15-Minute Coverage:** Automated digests every 15 minutes (Hourly at :00, 30-min summaries at :15 and :45).
*   **Midnight Recap:** A consolidated summary of the 02:00–05:00 WIB "sleep window" delivered at 05:00 AM.
*   **Spike Detection:** Triggers an immediate AI summary if message volume suddenly exceeds 30 msgs/min (e.g., during a Flash Sale).
*   **Thread Ramai:** Detects and summarizes trending discussions with AI, explaining exactly *why* the group is buzzing.

### ⚡ Performance Optimized
*   **High-Speed Engine:** Permanent processing loop clears message batches almost instantly.
*   **Lean Database:** 24-hour rolling retention keeps the SQLite database fast and relevant.
*   **VPS Friendly:** Optimized for 1GB RAM machines with low CPU overhead and WAL-mode database stability.

## 🛠 Tech Stack
*   **Core:** Python 3.11+
*   **Telegram:** Telethon (Listener) & Python-Telegram-Bot (Interface)
*   **AI:** Google GenAI (Gemma-2.5/Gemma-4 models)
*   **Database:** SQLite (WAL Mode) with `aiosqlite`
*   **Scheduling:** APScheduler

## 📋 Commands (Owner Only)
*   `/summary [jam]` — List promotions from the last X hours.
*   `/today` — Accurate chronological log of today's deals (WIB).
*   `/aisummary [n]` — Ask AI to summarize the last N raw messages.
*   `/status` — Check queue health, database size, and system vitals.
*   `/ping` — Verify uptime and connection status.
*   `/restart` — Safe service restart.

## 🏗 Setup & Deployment
The bot is designed to run as a `systemd` service on a Linux VPS.

1.  **Environment:** Copy `.env.example` to `.env` and fill in your API keys.
2.  **Service:** Use the provided `tanyadfbot.service` file for auto-restart and boot persistence.
3.  **Optimization:** On low-RAM instances, it is recommended to disable the Google Cloud Ops Agent to free up ~100MB of memory.

---
*Built with ❤️ for the Discountfess community.*
