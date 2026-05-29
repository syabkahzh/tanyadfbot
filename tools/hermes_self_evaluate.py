#!/usr/bin/env python3
"""hermes_self_evaluate.py — Ground-truth evaluation + auto-refinement.

Compares what TanyaDFBot extracted (promos table) against actual messages
(messages table) to find:
  1. Missed promos — messages that look like promos but weren't captured
  2. Wrong brands — brand mismatches between raw text and extracted brand
  3. False positives — complaints/chat captured as promos
  4. Bad summaries — too short, vague, junk
  5. Status mismatches — raw text says expired but status=active

Auto-fixes:
  - Updates _BRAND_CANON for new brand variants
  - Updates _COMPLAINT_PATTERN for new complaint keywords
  - Updates _JUNK_SUMMARIES for new junk patterns
  - Fixes status in DB when clearly wrong

Usage:
    PYTHONPATH=. .venv/bin/python tools/hermes_self_evaluate.py --hours 1
"""

import sqlite3
import re
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta

DB_PATH = Path(__file__).parent.parent / "tanya_main.db"
WIB = timezone(timedelta(hours=7))

# ── Known patterns (mirrors processor.py) ───────────────────────────────
_STRONG_KEYWORDS = {
    'sfood','gfood','grab','shopee','gojek','tokped','tokopedia',
    'voucher','vcr','voc','diskon','promo','cashback','gratis','potongan',
    'idm','indomaret','alfa','alfamart','alfagift','hokben',
    'klaim','claim','restock','ristok','nt','abis','habis',
    'gabisa','gaada','g+b+s','gamau','minbel',
    'kuota','limit','slot','redeem','qr','scan','edc',
    'cb','kesbek','c+s+h+b+c+k','cash back',
    'luber','pecah','flash','sale','deal','murah','hemat','bonus',
    'ongkir','gratis ongkir','membership','member','mamber',
    'tukpo','murce','murmer','sopi','tsel','cgv','xxi',
    'svip','badut','war','begal','kreator',
    'kopken','chatime','gindaco','solaria','rotio','spx','gopay','spay',
    'shopeepay','ovo','neo','tmrw','saqu','seabank','hero',
    'blibli','serbabu','famima','familymart','supin','superindo',
    'gacoan','mie gacoan','wingstop','yoshinoya','azko','sei',
    'flip','superbank','dana','tts','emados','ndog','pc','garap',
}

_QUESTION_PATTERN = re.compile(
    r'(\?|apa\s+ini|gimana\s+caranya|brp\s+harganya|berapa\s+harga|'
    r'kapan\s+mulai|dimana\s+bisa|kenapa\s+|ada\s+yg\s+tau|'
    r'pake\s+voucher\s+apa|pake\s+voc\s+apa|'
    r'cara\s+claim|cara\s+klaim|cara\s+pakai|'
    r'gmn|gmana|gmna|nta|tanya|nanya|'
    r'(?:masih|lagi)\s+(?:ada|on|work|aktif)\s+(?:ga|gak|nggak|ya|yah|aja)|'
    r'(?:dpt|dapet|dapat|dapatkan)\s+\w*\s*(?:dari|dr)\s+\w|'
    r'brp\b|berapa\b|dpt\s+brp|'
    r'worth\s+(?:it|ga|gak|ya|yah)|'
    r'(?:stay|ada)\s+(?:dimana|dmn|mana|gimana|gmn)|'
    r'(?:masih|udah|belum)\s+\w+\s*(?:ga|gak|ya|yah)|'
    r'(?:lg|lagi)\s+\w+\s*(?:ga|gak|ya|yah)|'
    r'(?:bisa|bs)\s+\w+\s+(?:ga|gak|gabisa|nggak|ga bisa)|'
    r'(?:bisa|bs)\s+(?:ga|gak|gabisa|nggak|ga bisa)|'
    r'gimana\s+(?:sih|dong|kak|ka|gy)|'
    r'gmn\s+sih|ko\s+tumben|'
    r'(?:pada|pade)\s+dpt|'
    r'yang\s+(?:tau|tahu|tw)\s+\w|'
    r'(?:masih|lagi)\s+ada.*(?:butuh|pake|pakai)|'
    r'gimana\s+ya\s+caranya|'
    r'ada\s+yg\s+dpt\s+kah|'
    r'mending\s+(?:komplain|ga|gak|nggak)|'
    r'jam\s+\d+.*masih\s+ada|'
    r'(?:apa|knp|kenapa)\s+(?:ngga|nggak|ga|gak)\s+ya|'
    r'(?:ada\s+opsi|ada\s+cara)|'
    r'yang\s+kena\b|'
    r'susah\s+kalo|'
    r'saranin\s+\w+|'
    r'(?:apa|gimana)\s+ngga\s+ya)',
    re.IGNORECASE
)

_COMPLAINT_PATTERN = re.compile(
    r'(gangguan|ga bisa|gak bisa|gk bisa|ga jalan|gak jalan|rusak|error|bug|'
    r'habis terus|ga dapet|gak dapet|ga keluar|gak keluar|'
    r'belum on|belum aktif|belum jalan|masih off|'
    r'kena refund|dibatalkan|dicancel|'
    r'tiba2|tiba-tiba|tiba tiba|'
    r'tampilan|interface|ui|layar|tampil|gimana ini|refresh|'
    r'masalah|trouble|ganggu|'
    r'balik ke|ga mau|gak mau|gamau|mau.*nempel|'
    r'gk ada|ga ada|gak ada)',
    re.IGNORECASE
)

_BOIKOT_PATTERN = re.compile(
    r'(paylater|pay\s*later|cicilan|kredit|bank\s*saqu|saqu|superbank)',
    re.IGNORECASE
)

_SOCIAL_FILLER = re.compile(
    r'^(?:wkwk|haha|hehe|iya|noted|oke|ok|makasih|thanks|thx|mantap|gas|bos|guys|gais|bang|kak|siap|sip|lol|anjir|anjay|btw|oot|gws|semangat|ya allah|nangis|sedih|beneran|kah|[!.\s])+$',
    re.IGNORECASE
)

# ── Brand canon (subset for matching) ───────────────────────────────────
_BRAND_CANON_KEYS = {
    'shopee', 'shopeefood', 'sfood', 'gojek', 'gofood', 'gfood',
    'gopay', 'grab', 'tokopedia', 'tokped', 'lazada', 'lzd',
    'indomaret', 'idm', 'alfamart', 'alfa', 'jsm', 'psm', 'afm',
    'telkomsel', 'tsel', 'ovo', 'dana', 'shopeepay', 'spay',
    'kopi kenangan', 'kopken', 'chatime', 'cgv', 'xxi',
    'hokben', 'solaria', 'wingstop', 'mcd', 'kfc',
    'ultravoucher', 'uv deals', 'u+v',
    'blibli', 'tokopedia', 'bukalapak',
    'seabank', 'neo', 'tmrw', 'jago',
    'pln', 'gofood', 'spx', 'alfagift', 'ag',
    'maybank', 'bca', 'bri', 'bni', 'mandiri',
    'halodoc', 'tokocash', 'myvaku',
    'hop hop', 'hophop',
    'spud', 'skin1004', 'aice',
    'tiket', 'tiket.com', 'traveloka',
    'zalora', 'bukalapak', 'blibli',
    'grabfood', 'grabfood', 'grab',
    'sopi', 'sopifood',
    'rotio', 'roti o',
    'tomoro', 'tomoro coffee',
    'point coffee', 'poin coffee',
    'gindaco', 'cupbop',
    'kawanlama', 'kawan lama',
    'pc hemat', 'pchematapril',
    'fortklass', 'g2g', 'burek',
    'bank neo', 'bank jago', 'bank aladin',
    'mybca', 'astrapay', 'alt',
    'mayan',
}


def _cutoff_iso(hours: int) -> str:
    """Return cutoff timestamp in the same format the DB stores (no T, no μs)."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return cutoff.strftime("%Y-%m-%d %H:%M:%S+00:00")


def get_recent_messages(hours: int = 1) -> list[dict]:
    """Fetch recent messages from the group."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, tg_msg_id, text, timestamp, processed, sender_name,
                  skip_reason
           FROM messages
           WHERE timestamp > ?
           ORDER BY timestamp ASC""",
        (_cutoff_iso(hours),)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_promos(hours: int = 1) -> list[dict]:
    """Fetch recent promos extracted by the bot."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, source_msg_id, summary, brand, status, conditions,
                  created_at
           FROM promos
           WHERE created_at > ?
           ORDER BY created_at ASC""",
        (_cutoff_iso(hours),)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


_CASUAL_CHAT = re.compile(
    r'^(?:beli\s+\w+|makan\s+\w+|lagi\s+\w+|habis\s+\w+|'
    r'udah\s+\w+|baru\s+\w+|mau\s+\w+|'
    r'kemarin\s+\w+|kemaren\s+\w+|'
    r'gak\s+\w+|ga\s+\w+|nggak\s+\w+|'
    r'(?:iya|oh|ok|oke|nah|hehe|haha|wkwk|sip|mantap|jos|alhamdulillah))',
    re.IGNORECASE
)

_REPLY_PATTERN = re.compile(
    r'^(?:iya|oh|ok|oke|nah|hehe|haha|wkwk|sip|mantap|jos|'
    r'ga\s+tau|gak\s+tau|gatau|alhamdulillah|'
    r'makasih|thanks|thx|mks|terimakasih|'
    r'on|off|aktif|habis|abis| expired|'
    r'(?:bisa|bs)\s+(?:ga|gak|ya)|'
    r'(?:masih|udah|belum)\s+\w+)',
    re.IGNORECASE
)


def classify_message(text: str) -> str:
    """Classify a raw message into category."""
    if not text or len(text.strip()) < 3:
        return "too_short"
    t = text.strip()
    if _SOCIAL_FILLER.match(t):
        return "social_filler"
    if _BOIKOT_PATTERN.search(t):
        return "boikot"
    if _COMPLAINT_PATTERN.search(t):
        return "complaint"
    if _QUESTION_PATTERN.search(t):
        return "question"
    if _REPLY_PATTERN.match(t):
        return "reply"
    if _CASUAL_CHAT.match(t):
        return "casual_chat"
    # Check for strong promo keywords — but require multi-word or action context
    words = set(re.findall(r'\w+', t.lower()))
    strong_hits = words & _STRONG_KEYWORDS
    if strong_hits:
        # Require either: multiple strong keywords, or a price/percentage/discount context
        has_price = bool(re.search(r'rp\s?\d|rb\s?\d|\d+[kK]|\d+\s*%|cashback|diskon|voucher|gratis', t, re.IGNORECASE))
        has_action = bool(re.search(r'(?:klaim|claim|restock|aktif|on|work|nyala|habis|expired|nt|diskon|cashback|bonus|gratis|potongan|luber|pecah|flash|sale|deal)', t, re.IGNORECASE))
        if len(strong_hits) >= 2 or has_price or has_action:
            return "likely_promo"
    return "unknown"


def find_brand_in_text(text: str) -> str | None:
    """Try to extract a brand name from raw text."""
    t = text.lower()
    for key in sorted(_BRAND_CANON_KEYS, key=len, reverse=True):
        if key in t:
            return key
    return None


def evaluate(messages: list[dict], promos: list[dict]) -> dict:
    """Compare messages vs promos to find issues."""
    promo_by_msg = {p["source_msg_id"]: p for p in promos if p.get("source_msg_id")}
    msg_ids_with_promo = set(promo_by_msg.keys())

    results = {
        "total_messages": len(messages),
        "total_promos": len(promos),
        "coverage": 0,
        "missed_promos": [],
        "false_positives": [],
        "brand_mismatches": [],
        "bad_summaries": [],
        "status_mismatches": [],
        "new_brand_variants": [],
        "new_complaint_keywords": [],
        "auto_fixes_applied": [],
        "by_category": Counter(),
    }

    # Analyze each message
    for msg in messages:
        mid = msg["id"]
        text = msg.get("text", "") or ""
        cat = classify_message(text)
        results["by_category"][cat] += 1

        # Missed promo: message classified as likely_promo but not captured
        if cat == "likely_promo" and mid not in msg_ids_with_promo:
            brand = find_brand_in_text(text)
            results["missed_promos"].append({
                "msg_id": mid,
                "tg_msg_id": msg.get("tg_msg_id"),
                "text": text[:200],
                "detected_brand": brand,
            })

        # False positive: complaint/filler/boikot captured as promo
        if cat in ("complaint", "social_filler", "boikot", "question", "too_short") and mid in msg_ids_with_promo:
            p = promo_by_msg[mid]
            results["false_positives"].append({
                "msg_id": mid,
                "tg_msg_id": msg.get("tg_msg_id"),
                "text": text[:200],
                "category": cat,
                "promo_summary": p.get("summary", ""),
                "promo_brand": p.get("brand", ""),
            })

    # Analyze captured promos
    for p in promos:
        summary = p.get("summary", "") or ""
        brand = p.get("brand", "") or ""
        status = p.get("status", "") or ""
        msg_id = p.get("source_msg_id")

        # Bad summary: too short or junk-like
        clean = re.sub(r'\*\*[^*]+\*\*\s*', '', summary).strip()
        if len(clean) < 8 and clean.lower() not in ('active', 'expired', 'unknown'):
            results["bad_summaries"].append({
                "promo_id": p["id"],
                "msg_id": msg_id,
                "summary": summary,
                "brand": brand,
            })

        # Brand in "Unknown" when we can detect one from DB lookup
        if brand == "Unknown" and msg_id:
            msg = next((m for m in messages if m["id"] == msg_id), None)
            if msg:
                detected = find_brand_in_text(msg.get("text", "") or "")
                if detected:
                    results["brand_mismatches"].append({
                        "promo_id": p["id"],
                        "msg_id": msg_id,
                        "current_brand": brand,
                        "detected_brand": detected,
                        "text": (msg.get("text", "") or "")[:200],
                    })

    # Coverage
    captured = len(promos)
    total_likely = results["by_category"].get("likely_promo", 0) + captured
    results["coverage"] = round(captured / max(total_likely, 1) * 100, 1)

    return results


def auto_fix_brand_canon(new_variants: list[dict]) -> list[str]:
    """Add new brand variants to _BRAND_CANON in db.py."""
    fixes = []
    db_path = Path(__file__).parent.parent / "db.py"
    content = db_path.read_text()

    for v in new_variants:
        variant = v["detected_brand"].lower()
        canonical = v["detected_brand"].title()
        # Check if already exists
        if f"'{variant}'" in content:
            continue
        # Find insertion point (after last brand entry)
        insert_line = f"    '{variant}': '{canonical}',"
        # Insert before the closing }
        if insert_line not in content:
            # Find the last entry before the closing brace
            marker = "    # junk sentinels → Unknown"
            if marker in content:
                content = content.replace(marker, insert_line + "\n" + marker)
                fixes.append(f"Added '{variant}' → '{canonical}' to _BRAND_CANON")

    if fixes:
        db_path.write_text(content)
    return fixes


def auto_fix_complaint_patterns(keywords: list[str]) -> list[str]:
    """Add new complaint keywords to _COMPLAINT_PATTERN in processor.py."""
    fixes = []
    proc_path = Path(__file__).parent.parent / "processor.py"
    content = proc_path.read_text()

    for kw in keywords:
        if kw.lower() in content.lower():
            continue
        # Find the complaint pattern block
        marker = "_COMPLAINT_PATTERN = re.compile("
        if marker in content:
            # Find the closing paren of the pattern
            idx = content.index(marker)
            # Find the last string line before re.IGNORECASE
            closing = content.index("re.IGNORECASE", idx)
            # Insert before closing
            insert = f"    r'{kw}|'\n"
            content = content[:closing] + insert + content[closing:]
            fixes.append(f"Added '{kw}' to _COMPLAINT_PATTERN")

    if fixes:
        proc_path.write_text(content)
    return fixes


def auto_fix_junk_summaries(summaries: list[str]) -> list[str]:
    """Add new junk patterns to _JUNK_SUMMARIES in processor.py."""
    fixes = []
    proc_path = Path(__file__).parent.parent / "processor.py"
    content = proc_path.read_text()

    for s in summaries:
        clean = s.strip().lower()
        if f"'{clean}'" in content:
            continue
        # Find _JUNK_SUMMARIES set
        marker = "_JUNK_SUMMARIES: set[str] = {"
        if marker in content:
            idx = content.index(marker)
            closing = content.index("}", idx)
            insert = f"    '{clean}',\n"
            content = content[:closing] + insert + content[closing:]
            fixes.append(f"Added '{clean}' to _JUNK_SUMMARIES")

    if fixes:
        proc_path.write_text(content)
    return fixes


def fix_status_in_db(promos_to_fix: list[dict]) -> int:
    """Fix status mismatches in the database."""
    if not promos_to_fix:
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    count = 0
    for p in promos_to_fix:
        conn.execute(
            "UPDATE promos SET status = 'expired' WHERE id = ?",
            (p["promo_id"],)
        )
        count += 1
    conn.commit()
    conn.close()
    return count


def format_wib(ts: str | None) -> str:
    """Format UTC timestamp to WIB string."""
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts.replace("+00:00", "+00:00"))
        wib = dt.astimezone(WIB)
        return wib.strftime("%H:%M WIB")
    except Exception:
        return ts[:16]


def generate_report(results: dict, fixes: list[str]) -> str:
    """Generate a concise evaluation report."""
    lines = []
    now = datetime.now(WIB)
    lines.append(f"📊 **Self-Eval Report** — {now.strftime('%d %b %H:%M WIB')}")
    lines.append("")

    # Coverage
    cov = results["coverage"]
    icon = "🟢" if cov >= 90 else "🟡" if cov >= 70 else "🔴"
    lines.append(f"Coverage: {icon} {cov}% ({results['total_promos']}/{results['total_messages']} msgs)")

    # Category breakdown
    cats = results["by_category"]
    lines.append(f"Kategori: likely={cats.get('likely_promo',0)} complaint={cats.get('complaint',0)} filler={cats.get('social_filler',0)} question={cats.get('question',0)}")

    # Issues
    issues = []
    if results["missed_promos"]:
        issues.append(f"❌ {len(results['missed_promos'])} promo terlewat")
    if results["false_positives"]:
        issues.append(f"⚠️ {len(results['false_positives'])} false positive")
    if results["bad_summaries"]:
        issues.append(f"📝 {len(results['bad_summaries'])} summary jelek")
    if results["brand_mismatches"]:
        issues.append(f"🏷️ {len(results['brand_mismatches'])} brand salah")

    if issues:
        lines.append("")
        lines.append("Issues:")
        for i in issues:
            lines.append(f"• {i}")

    # Missed promos detail (max 5)
    if results["missed_promos"]:
        lines.append("")
        lines.append("Top missed:")
        for mp in results["missed_promos"][:5]:
            brand = mp.get("detected_brand") or "?"
            txt = mp["text"][:80].replace("\n", " ")
            lines.append(f"  • [{brand}] {txt}")

    # False positives detail (max 3)
    if results["false_positives"]:
        lines.append("")
        lines.append("False positives:")
        for fp in results["false_positives"][:3]:
            lines.append(f"  • [{fp['category']}] {fp['promo_summary'][:60]}")

    # Auto-fixes
    if fixes:
        lines.append("")
        lines.append("🔧 Auto-fixes:")
        for f in fixes:
            lines.append(f"  • {f}")

    if not issues and not fixes:
        lines.append("")
        lines.append("✅ No issues found. System healthy.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Self-evaluate TanyaDFBot")
    parser.add_argument("--hours", type=int, default=1, help="Lookback window")
    parser.add_argument("--dry-run", action="store_true", help="Skip auto-fixes")
    args = parser.parse_args()

    messages = get_recent_messages(args.hours)
    promos = get_recent_promos(args.hours)

    results = evaluate(messages, promos)

    fixes = []
    if not args.dry_run:
        # Auto-fix brand canon
        if results["brand_mismatches"]:
            fixes.extend(auto_fix_brand_canon(results["brand_mismatches"]))

        # Auto-fix junk summaries
        if results["bad_summaries"]:
            junk = [b["summary"] for b in results["bad_summaries"] if len(b["summary"]) < 15]
            fixes.extend(auto_fix_junk_summaries(junk))

    report = generate_report(results, fixes)
    print(report)

    # Output structured data for Hermes agent
    if results["missed_promos"] or results["false_positives"]:
        print("\n---STRUCTURED_DATA---")
        print(json.dumps({
            "missed": len(results["missed_promos"]),
            "false_positives": len(results["false_positives"]),
            "brand_issues": len(results["brand_mismatches"]),
            "bad_summaries": len(results["bad_summaries"]),
            "fixes": len(fixes),
        }, indent=2))


if __name__ == "__main__":
    main()
