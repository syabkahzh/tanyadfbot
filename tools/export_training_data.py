import sqlite3
import re
import random
import os

def clean_text(text):
    if not text: return ""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r'\s+', ' ', text).lower()

def export_v2():
    # Robust path discovery
    if os.path.exists("tanya_main.db"):
        db_path = "tanya_main.db"
    elif os.path.exists("../tanya_main.db"):
        db_path = "../tanya_main.db"
    else:
        # Fallback to current config or env
        db_path = "tanya_main.db"

    output_path = "data/training.txt"
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print(f"📡 Exporting High-Fidelity dataset from {db_path}...")

    # 1. Fetch ALL confirmed promos
    cur.execute("""
        SELECT DISTINCT m.text FROM messages m
        JOIN promos p ON m.id = p.source_msg_id
        WHERE m.text IS NOT NULL AND length(m.text) > 4
    """)
    promos = [clean_text(row['text']) for row in cur.fetchall()]

    # 1c. Fetch PROMO SUMMARIES (Often formal/structured text)
    cur.execute("""
        SELECT DISTINCT summary FROM promos 
        WHERE summary IS NOT NULL AND length(summary) > 10
    """)
    summaries = [clean_text(row['summary']) for row in cur.fetchall()]
    
    # 1b. Fetch USER CORRECTIONS (Ground Truth - Force into PROMO)
    cur.execute("""
        SELECT DISTINCT m.text FROM messages m
        JOIN ai_corrections c ON m.id = c.original_msg_id
        WHERE m.text IS NOT NULL AND length(m.text) > 4
    """)
    corrections = [clean_text(row['text']) for row in cur.fetchall()]
    
    # Add everything to promos and remove duplicates
    promos = list(set(promos + summaries + corrections))

    # 1d. Manual SEED DATA (Force-feeding the model specific logic)
    seed_path = "data/seed_data.txt"
    seeds = []
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__label__PROMO "):
                    seeds.append(clean_text(line.replace("__label__PROMO ", "")))
    
    promos = list(set(promos + seeds))

    # 2. Fetch HIGH-QUALITY JUNK (Dynamic Discovery)
    # We find where we started recording skip reasons to avoid hardcoding IDs.
    cur.execute("SELECT MIN(id) FROM messages WHERE skip_reason IS NOT NULL AND skip_reason != ''")
    row = cur.fetchone()
    modern_start_id = row[0] if (row and row[0]) else 0
    
    # If the DB is fresh, ensure we at least look back far enough
    if modern_start_id == 0:
        cur.execute("SELECT MAX(id) FROM messages")
        max_id = cur.fetchone()[0] or 0
        modern_start_id = max(0, max_id - 10000)

    print(f"🧹 Fetching verified JUNK from modern range (ID >= {modern_start_id})...")
    cur.execute("""
        SELECT DISTINCT text FROM messages 
        WHERE id >= ? 
        AND skip_reason IN ('ai_skip', 'triage', 'regex', 'fasttext')
        AND text IS NOT NULL AND length(text) > 4
        AND id NOT IN (SELECT original_msg_id FROM ai_corrections)
    """, (modern_start_id,))
    modern_noise = [clean_text(row['text']) for row in cur.fetchall()]

    # 3. Fetch Legacy REGEX Noise (to maintain volume)
    cur.execute("""
        SELECT DISTINCT text FROM messages 
        WHERE (skip_reason = 'regex' OR skip_reason IS NULL)
        AND id < ?
        AND text IS NOT NULL AND length(text) > 4
    """, (modern_start_id,))
    legacy_noise = [clean_text(row['text']) for row in cur.fetchall()]


    conn.close()

    # BALANCING LOGIC
    # We prioritize modern_noise (verified by logic/AI) over legacy samples.
    final_noise = modern_noise + random.sample(legacy_noise, min(len(legacy_noise), 2000))

    
    # Shuffle
    random.shuffle(promos)
    random.shuffle(final_noise)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in promos:
            f.write(f"__label__PROMO {p}\n")
        for j in final_noise:
            f.write(f"__label__JUNK {j}\n")

    print(f"✅ Exported {len(promos)} PROMOs and {len(final_noise)} JUNK samples.")
    print(f"📊 Ratio: 1:{len(final_noise)/len(promos):.1f} (Ideal for preventing false positives)")

if __name__ == "__main__":
    export_v2()
