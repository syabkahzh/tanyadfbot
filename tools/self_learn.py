#!/usr/bin/env python3
"""Self-learning script: reads Discountfess group, detects new slang/terms/patterns.

Run periodically via Hermes cron. Outputs findings as text for the agent to process.
"""

import asyncio, os, re, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter

WIB = timezone(timedelta(hours=7))

# Known glossary — update this as new terms are confirmed
KNOWN_GLOSSARY = {
    'getok', 'ndog', 'telur', 'buang telur', 'kombo', 'kombo', 'vc', 'voc',
    'laz', 'tsel', 'f01', 'sfood', 'gfood', 'sopi', 'spay', 'spx',
    'alfa', 'jsm', 'psm', 'afm', 'idm', 'kopken', 'cgv', 'xxi', 'svip',
    'tukpo', 'minbel', 'gratong', 'jp', 'nt', 'koid', 'zonk', 'badut',
    'cb', 'kesbek', 'fs', 'ag', 'alfagift', 'neo', 'tmrw', 'saqu',
    'seabank', 'hero', 'blibli', 'famima', 'supin', 'superindo',
    'gacoan', 'wingstop', 'yoshinoya', 'azko', 'sei', 'flip', 'superbank',
    'dana', 'tts', 'emados', 'garap', 'serbabu', 'goib', 'begal', 'kreator',
    'pc', 'ndog', 'luber', 'pecah', 'war',
    # Common slang
    'mantap', 'gaskeun', 'woles', 'cupu', 'anjay', 'anjir', 'btw',
}

# Promo mechanism keywords
MECHANISM_KEYWORDS = [
    'klaim', 'claim', 'restock', 'ristok', 'flash sale', 'fs',
    'limit', 'slot', 'redeem', 'qr', 'scan', 'edc',
    'minbel', 'minimum belanja', 'cashback', 'potongan', 'diskon',
    'gratis ongkir', 'gratong', 'voucher', 'koin', 'poin',
    'tukpo', 'tukar poin', 'kombo', 'combo', 'b1g1', 'buy 1 get 1',
]


async def main():
    # Load env
    env_path = Path.home() / "telegram-mcp" / ".env"
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    from telethon import TelegramClient
    from telethon.sessions import StringSession

    client = TelegramClient(
        StringSession(os.environ["TELEGRAM_SESSION_STRING"]),
        int(os.environ["TELEGRAM_API_ID"]),
        os.environ["TELEGRAM_API_HASH"]
    )
    await client.connect()
    entity = await client.get_entity("discountfessofficialbase")

    # Read last 6 hours
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=6)

    messages = []
    async for msg in client.iter_messages(entity, offset_date=now, limit=300):
        if msg.date < since:
            break
        if msg.text:
            messages.append(msg)

    await client.disconnect()

    if not messages:
        print("📭 No messages in last 6 hours.")
        return

    # Analysis
    all_text = " ".join(m.text.lower() for m in messages if m.text)

    # 1. Detect potential new slang (repeated 2+ times, not in known glossary)
    words = re.findall(r'\b[a-z]{2,15}\b', all_text)
    word_counts = Counter(words)
    new_terms = {
        w: c for w, c in word_counts.items()
        if c >= 3 and w not in KNOWN_GLOSSARY
        and w not in {'yang', 'dan', 'ini', 'itu', 'untuk', 'dengan', 'ada',
                      'bisa', 'untuk', 'dari', 'juga', 'sudah', 'belum',
                      'masih', 'lagi', 'kak', 'ka', 'guys', 'gais', 'ges',
                      'gaes', 'bun', 'mba', 'mas', 'bang', 'bro', 'sis',
                      'iya', 'ok', 'oke', 'sip', 'siap', 'noted', 'sama',
                      'mau', 'di', 'ke', 'dia', 'nya', 'kan', 'sih', 'nih',
                      'deh', 'dong', 'yah', 'ya', 'lah', 'kah', 'nih',
                      'eh', 'woi', 'bro', 'cuy', 'gan', 'agan'}
    }

    # 2. Detect promo mechanisms mentioned
    mechanisms = {}
    for kw in MECHANISM_KEYWORDS:
        count = all_text.count(kw.lower())
        if count > 0:
            mechanisms[kw] = count

    # 3. Detect time-sensitive mentions (deadline, hari ini, besok, etc.)
    time_keywords = re.findall(
        r'(hari ini|besok|lusa|deadline|expir|habis|last day|reset|'
        r'tanggal \d+|sampai \d+|sdh \d+|before \d+)',
        all_text
    )
    time_mentions = Counter(time_keywords)

    # 4. User activity summary
    user_counts = Counter()
    for msg in messages:
        if msg.sender:
            name = getattr(msg.sender, 'first_name', '') or 'Unknown'
            user_counts[name] += 1

    # Output
    print(f"📊 Self-Learning Report ({len(messages)} messages, last 6h)")
    print(f"⏰ {datetime.now(WIB).strftime('%Y-%m-%d %H:%M WIB')}")
    print()

    if new_terms:
        print("🆕 Potential New Terms:")
        for term, count in sorted(new_terms.items(), key=lambda x: -x[1])[:10]:
            print(f"  • {term} (×{count})")
        print("  → Review & add to glossary if they're real slang")
        print()

    if mechanisms:
        print("⚙️ Promo Mechanisms Detected:")
        for kw, count in sorted(mechanisms.items(), key=lambda x: -x[1])[:8]:
            print(f"  • {kw} (×{count})")
        print()

    if time_mentions:
        print("⏰ Time-Sensitive Mentions:")
        for kw, count in sorted(time_mentions.items(), key=lambda x: -x[1])[:5]:
            print(f"  • {kw} (×{count})")
        print()

    print("👥 Top Active Users:")
    for name, count in user_counts.most_common(5):
        print(f"  • {name}: {count} msgs")

    # Save raw findings for agent
    findings = {
        'timestamp': datetime.now(WIB).isoformat(),
        'message_count': len(messages),
        'new_terms': dict(sorted(new_terms.items(), key=lambda x: -x[1])[:10]),
        'mechanisms': dict(sorted(mechanisms.items(), key=lambda x: -x[1])[:8]),
        'time_mentions': dict(sorted(time_mentions.items(), key=lambda x: -x[1])[:5]),
        'top_users': dict(user_counts.most_common(5)),
    }
    findings_path = Path.home() / "tanyadfbot" / "tools" / "last_learning.json"
    findings_path.write_text(json.dumps(findings, indent=2, ensure_ascii=False))
    print(f"\n💾 Findings saved to {findings_path}")


if __name__ == "__main__":
    asyncio.run(main())
