import sqlite3
import os

def label_data(db_path="tanya_main_vps.db"):
    if not os.path.exists(db_path):
        print(f"❌ Error: {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("📊 Analyzing database...")

    # 1. Ensure the column exists (just in case)
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN skip_reason TEXT;")
        print("✅ Added skip_reason column.")
    except sqlite3.OperationalError:
        print("ℹ️ skip_reason column already exists.")

    # 2. Tag HIGH-QUALITY NOISE
    # Logic: If processed=1 but NOT in the promos table, it's a negative example.
    print("🧹 Tagging noise (negative examples)...")
    cursor.execute("""
        UPDATE messages 
        SET skip_reason = 'ai_skip' 
        WHERE processed = 1 
        AND id NOT IN (
            SELECT source_msg_id FROM promos WHERE source_msg_id IS NOT NULL
        );
    """)
    noise_count = cursor.rowcount

    # 3. Clean PROMOS (positive examples)
    # Logic: Ensure promos have no skip_reason so they are identified as the positive class.
    cursor.execute("""
        UPDATE messages 
        SET skip_reason = NULL 
        WHERE id IN (
            SELECT source_msg_id FROM promos WHERE source_msg_id IS NOT NULL
        );
    """)
    promo_count = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"\n✨ Done! Ready for FastText training:")
    print(f"✅ {promo_count} Positive Examples (Promos)")
    print(f"❌ {noise_count} Negative Examples (Noise)")
    print(f"\nNext: Run your export script to generate the FastText .txt file!")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tanya_main_vps.db"
    label_data(path)
