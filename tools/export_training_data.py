import sqlite3
import re
import random

def clean_text(text):
    if not text: return ""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r'\s+', ' ', text).lower()

def export_v2():
    db_path = "../tanya_main.db"
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

    # 1b. Fetch USER CORRECTIONS (Ground Truth - Force into PROMO)
    cur.execute("""
        SELECT DISTINCT m.text FROM messages m
        JOIN ai_corrections c ON m.id = c.original_msg_id
        WHERE m.text IS NOT NULL AND length(m.text) > 4
    """)
    corrections = [clean_text(row['text']) for row in cur.fetchall()]
    
    # Add corrections to promos and remove duplicates
    promos = list(set(promos + corrections))

    # 2. Fetch HIGH-QUALITY Noise (messages the AI actually looked at but skipped)
    # CRITICAL: Exclude messages that were later corrected by user!
    cur.execute("""
        SELECT DISTINCT m.text FROM messages m
        WHERE m.skip_reason = 'ai_skip' 
        AND m.text IS NOT NULL AND length(m.text) > 4
        AND m.id NOT IN (SELECT original_msg_id FROM ai_corrections)
    """)
    ai_noise = [clean_text(row['text']) for row in cur.fetchall()]

    # 3. Fetch REGEX-FILTERED Noise (low-level chatter)
    cur.execute("""
        SELECT DISTINCT text FROM messages 
        WHERE skip_reason = 'regex' 
        AND text IS NOT NULL AND length(text) > 4
    """)
    regex_noise = [clean_text(row['text']) for row in cur.fetchall()]

    conn.close()

    # BALANCING LOGIC
    # We want plenty of noise so the model isn't trigger-happy.
    # Total JUNK = AI-skipped (high value) + Sampled Regex-skipped
    final_noise = ai_noise + random.sample(regex_noise, min(len(regex_noise), 3000))
    
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
