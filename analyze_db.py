import sqlite3
import collections
import re

db_path = './data/tanya_main.db'

def get_db():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def analyze_chat():
    conn = get_db()
    cur = conn.cursor()

    print("--- 📊 TANYA DF CHAT ANALYSIS ---")
    
    # Total messages
    cur.execute("SELECT COUNT(*) FROM messages")
    total_msgs = cur.fetchone()[0]
    print(f"Total Messages: {total_msgs}")
    
    # 1. Message Length Distribution
    cur.execute("SELECT length(text) FROM messages WHERE text IS NOT NULL")
    lengths = [r[0] for r in cur.fetchall()]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    short_msgs = sum(1 for l in lengths if l < 20)
    print(f"Average Message Length: {avg_len:.1f} chars")
    print(f"Short Messages (<20 chars): {short_msgs} ({(short_msgs/total_msgs)*100:.1f}%)")

    # 2. Slang & Keyword Frequencies
    stop_words = {'yang','yg','iya','gak','ada','aku','kak','juga','gais','bisa','ga','ya','di','ke','dan','atau','tapi','pls','nih','ini','itu','udah','dah','nggk','lagi','aja','kok','kalo','smpe','dri','dr','tdk','blm','mau','tanya','tau','lho','sih','dong','deh','banget','emang','kayak','terus','jadi','sama','kaya','punya','abis','habis','dengan','untuk','dari','buat','biasa','kalau','kayaknya','pake','boleh','nuker','beli','tadi','masuk','minta','coba','kasih', 'nya', 'kalian', 'sekarang'}
    cur.execute("SELECT text FROM messages WHERE text IS NOT NULL")
    words = collections.Counter()
    bigrams = collections.Counter()
    
    for row in cur.fetchall():
        text = row[0].lower()
        tokens = [w for w in re.findall(r'\b[a-z]+\b', text) if len(w) > 2 and w not in stop_words]
        for w in tokens:
            words[w] += 1
        for i in range(len(tokens) - 1):
            bigrams[f"{tokens[i]} {tokens[i+1]}"] += 1
            
    print("\n--- 🔥 Top Promo Keywords / Slang ---")
    for w, count in words.most_common(20):
        print(f"{w}: {count}")
        
    print("\n--- 🗣️ Top Phrases (Bigrams) ---")
    for bg, count in bigrams.most_common(10):
        print(f"{bg}: {count}")

    # 3. Context Analysis around short signals (e.g. "aman")
    cur.execute("SELECT text FROM messages WHERE text LIKE '%aman%'")
    aman_msgs = cur.fetchall()
    brands_found = 0
    total_aman = len(aman_msgs)
    
    brands = ['hokben', 'hophop', 'shopeefood', 'sfood', 'gofood', 'kopken', 'spx', 'ismaya', 'kfc', 'mcd', 'alfamart', 'indomaret', 'cgv', 'xxi']
    for row in aman_msgs:
        text = row[0].lower()
        if any(b in text for b in brands):
            brands_found += 1
            
    print("\n--- 🧩 Context Dependency ('Aman' analysis) ---")
    print(f"Messages containing 'aman': {total_aman}")
    print(f"Of those, mention brand explicitly: {brands_found} ({(brands_found/total_aman*100 if total_aman else 0):.1f}%)")

    # 4. Question Analysis
    cur.execute("SELECT text FROM messages WHERE text LIKE '%?%'")
    total_questions = cur.fetchall()
    print("\n--- ❓ Question Frequency ---")
    print(f"Messages containing a question mark: {len(total_questions)} ({(len(total_questions)/total_msgs)*100:.1f}%)")

    conn.close()

if __name__ == '__main__':
    analyze_chat()