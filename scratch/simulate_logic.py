import asyncio
import sqlite3
from processor import GeminiProcessor
import shared

async def simulate():
    # Load processor
    gp = GeminiProcessor()
    await shared.load_classifier("model.ftz")
    
    # Connect to OLD database to get sample data
    conn = sqlite3.connect('data/tanya_main.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("SELECT id, text FROM messages WHERE text IS NOT NULL ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()
    
    results = []
    dropped_count_regex = 0
    dropped_count_ft = 0
    passed_count = 0
    
    for r in rows:
        text = r['text']
        # Tier 1: Regex
        worth_checking = gp._is_worth_checking(text)
        
        # Tier 2: FastText
        label, conf = await shared.classify(text)
        
        will_process = worth_checking and not (label == '__label__JUNK' and conf >= 0.98)
        
        res = {
            'id': r['id'],
            'text': text.replace('\n', ' ')[:100],
            'regex_worth': worth_checking,
            'ft_label': label,
            'ft_conf': conf,
            'will_process': will_process
        }
        results.append(res)
        
        if not worth_checking:
            dropped_count_regex += 1
        elif (label == '__label__JUNK' and conf >= 0.98):
            dropped_count_ft += 1
        else:
            passed_count += 1
    
    print("\n--- Filter Simulation Results ---")
    print(f"Total messages: {len(results)}")
    print(f"Dropped by Regex: {dropped_count_regex}")
    print(f"Dropped by FastText: {dropped_count_ft}")
    print(f"Passed to AI: {passed_count}")
    
    print("\n--- Examples of DROPPED messages ---")
    dropped = [r for r in results if not r['will_process']][:10]
    for r in dropped:
        reason = "Regex" if not r['regex_worth'] else f"FastText ({r['ft_label']} {r['ft_conf']:.2f})"
        print(f"[{reason}] {r['text']}")

    print("\n--- Examples of PASSED messages ---")
    passed = [r for r in results if r['will_process']][:10]
    for r in passed:
        print(f"[PASS] {r['text']}")

if __name__ == "__main__":
    asyncio.run(simulate())
