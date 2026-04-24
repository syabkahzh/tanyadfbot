import sqlite3
import re

def clean_text(text):
    if not text: return ""
    # Remove newlines and tabs
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Simple whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def export():
    db_path = "tanya_main_vps.db"
    output_path = "train_data.txt"
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print(f"📦 Exporting from {db_path} to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        # 1. Export PROMOS (Positives)
        cursor.execute("""
            SELECT text FROM messages 
            WHERE id IN (SELECT source_msg_id FROM promos WHERE source_msg_id IS NOT NULL)
        """)
        promos = cursor.fetchall()
        for r in promos:
            txt = clean_text(r['text'])
            if txt:
                f.write(f"__label__PROMO {txt}\n")

        # 2. Export NOISE (Negatives)
        cursor.execute("SELECT text FROM messages WHERE skip_reason = 'ai_skip'")
        noise = cursor.fetchall()
        for r in noise:
            txt = clean_text(r['text'])
            if txt:
                f.write(f"__label__NOISE {txt}\n")

    conn.close()
    print(f"✅ Success! {len(promos)} promos and {len(noise)} noise exported.")
    print(f"👉 File ready: {output_path}")

if __name__ == "__main__":
    export()
