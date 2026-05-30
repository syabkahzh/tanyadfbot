#!/usr/bin/env python3
"""
TanyaDFBot Maestro Dashboard
============================
Comprehensive analysis: gap detection, quality audit, health check.

Usage:
  cd ~/tanyadfbot && PYTHONPATH=. .venv/bin/python tools/maestro_dashboard.py --hours 24
  cd ~/tanyadfbot && PYTHONPATH=. .venv/bin/python tools/maestro_dashboard.py --hours 6
"""
import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

WIB = timezone(timedelta(hours=7))

# ─── Config ───────────────────────────────────────────────────────────
GROUP_ENTITY = "discountfessofficialbase"
GROUP_ID = 1852914963
TELEGRAM_MCP_DIR = Path.home() / "telegram-mcp"
DB_PATH = Path.home() / "tanyadfbot" / "tanya_main.db"

# Brand canon from db.py (subset for matching)
BRAND_CANON = {
    "shopee": "Shopee", "tokopedia": "Tokopedia", "lazada": "Lazada",
    "tiket": "Tiket.com", "tiketcom": "Tiket.com", "traveloka": "Traveloka",
    "grab": "Grab", "gojek": "Gojek", "dana": "DANA",
    "ovo": "OVO", "gopay": "GoPay", "qris": "QRIS",
    "bca": "BCA", "bri": "BRI", "bni": "BNI", "mandiri": "Mandiri",
    "mandiri": "Mandiri", "btn": "BTN", "bsi": "BSI", "bjb": "BJB",
    "xl": "XL", "xlaxiata": "XL Axiata", "axis": "Axis",
    "telkomsel": "Telkomsel", "indosat": "Indosat", "tri": "Tri",
    "smartfren": "Smartfren",
    "kai": "KAI", "kereta": "KAI",
    "popmie": "Pop Mie", "kredivo": "Kredivo", "homecredit": "Home Credit",
    "blibli": "Blibli", "bukalapak": "Bukalapak",
    "mybca": "MyBCA", "bca": "BCA",
}


def load_telethon_env():
    """Load env vars from telegram-mcp/.env"""
    env_path = TELEGRAM_MCP_DIR / ".env"
    if not env_path.exists():
        print(f"❌ No .env at {env_path}")
        sys.exit(1)
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


async def read_group_messages(hours):
    """Read messages from the group via Telethon."""
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    load_telethon_env()
    client = TelegramClient(
        StringSession(os.environ["TELEGRAM_SESSION_STRING"]),
        int(os.environ["TELEGRAM_API_ID"]),
        os.environ["TELEGRAM_API_HASH"]
    )
    await client.connect()

    entity = await client.get_entity(GROUP_ENTITY)
    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=hours)

    messages = []
    async for msg in client.iter_messages(entity, offset_date=now_utc, limit=500):
        if msg.date < since_utc:
            break
        messages.append({
            "id": msg.id,
            "text": msg.text or "",
            "date": msg.date,
            "sender": getattr(msg.sender, "first_name", "") or "",
            "sender_id": getattr(msg.sender, "id", None),
            "is_reply": bool(msg.reply_to),
        })

    await client.disconnect()
    messages.sort(key=lambda m: m["date"])
    return messages


def get_db_promos(hours):
    """Get promos from DB in the time window."""
    if not DB_PATH.exists():
        print(f"❌ DB not found at {DB_PATH}")
        return []
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        SELECT p.id, p.summary, p.brand, p.status, p.tg_link, p.created_at,
               p.source_msg_id, m.text as raw_text, m.timestamp as msg_time
        FROM promos p
        LEFT JOIN messages m ON p.source_msg_id = m.id
        WHERE p.created_at >= ?
        ORDER BY p.created_at DESC
    """, (since,))
    
    promos = [dict(row) for row in cur.fetchall()]
    conn.close()
    return promos


def get_db_messages(hours):
    """Get raw messages from DB in the time window."""
    if not DB_PATH.exists():
        return []
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        SELECT id, sender_name, text, timestamp
        FROM messages
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
    """, (since,))
    
    msgs = [dict(row) for row in cur.fetchall()]
    conn.close()
    return msgs


def detect_potential_promos(group_messages):
    """Heuristic scan for potential promo mentions in raw group messages."""
    promo_signals = []
    
    # Keywords that suggest a promo/deal
    deal_keywords = [
        "gratis", "free", "cashback", "diskon", "discount", "promo",
        "voucher", "vc ", "kode", "referral", "reff", "cashback",
        "hadiah", "bonus", "claim", "klaim", "gratisan", "serba",
        "10%", "20%", "30%", "50%", "75%", "90%",
        "rp ", "idr", "harga", "murah", "hemat",
        "limited", "terbatas", "ending", "last day",
        "shopee", "tokopedia", "lazada", "grab", "gojek",
        "mybca", "bca", "bri", "bni", "mandiri",
        "xl", "axis", "telkomsel", "indosat",
        "kai", "kereta", "tiket",
    ]
    
    for msg in group_messages:
        text = msg["text"].lower()
        if not text or len(text) < 5:
            continue
        
        # Skip very short messages
        if msg["is_reply"]:
            continue  # replies are usually discussion, not promos
        
        signals_found = [kw for kw in deal_keywords if kw in text]
        if len(signals_found) >= 2:  # At least 2 signals
            promo_signals.append({
                "id": msg["id"],
                "text": msg["text"][:200],
                "signals": signals_found,
                "date": msg["date"],
                "sender": msg["sender"],
            })
    
    return promo_signals


def gap_analysis(group_promos, db_promos):
    """Compare group activity with DB promos to find gaps."""
    # Get all source_msg_ids in DB
    db_msg_ids = set()
    for p in db_promos:
        if p.get("source_msg_id"):
            db_msg_ids.add(p["source_msg_id"])
    
    # Group messages that look like promos but NOT in DB
    missed = [p for p in group_promos if p["id"] not in db_msg_ids]
    
    return {
        "group_promo_signals": len(group_promos),
        "db_promos": len(db_promos),
        "missed_count": len(missed),
        "missed": missed[:20],  # cap at 20
    }


def quality_audit(db_promos):
    """Audit quality of existing promos."""
    issues = []
    
    for p in db_promos:
        summary = p.get("summary", "")
        brand = p.get("brand", "")
        status = p.get("status", "")
        raw_text = p.get("raw_text", "")
        
        # Check 1: Empty/very short summary
        if len(summary) < 10:
            issues.append({
                "promo_id": p["id"],
                "issue": "summary_too_short",
                "summary": summary[:100],
                "brand": brand,
            })
        
        # Check 2: Brand mismatch (summary mentions different brand)
        if brand and summary:
            summary_lower = summary.lower()
            brand_lower = brand.lower()
            if brand_lower not in summary_lower:
                # Check if another brand is mentioned
                for alt_brand in BRAND_CANON:
                    if alt_brand in summary_lower and alt_brand not in brand_lower:
                        issues.append({
                            "promo_id": p["id"],
                            "issue": "possible_brand_mismatch",
                            "summary": summary[:100],
                            "brand": brand,
                            "mentioned": alt_brand,
                        })
                        break
        
        # Check 3: Status mismatch with raw text
        if raw_text and status:
            raw_lower = raw_text.lower()
            if status == "active" and any(w in raw_lower for w in ["habis", "expired", "gaada", "ga ada", "udah abis"]):
                issues.append({
                    "promo_id": p["id"],
                    "issue": "status_mismatch",
                    "summary": summary[:100],
                    "brand": brand,
                    "status": status,
                    "raw_hint": raw_text[:100],
                })
            elif status == "expired" and any(w in raw_lower for w in ["masih aktif", "masih on", "masih bisa"]):
                issues.append({
                    "promo_id": p["id"],
                    "issue": "status_mismatch",
                    "summary": summary[:100],
                    "brand": brand,
                    "status": status,
                    "raw_hint": raw_text[:100],
                })
        
        # Check 4: Question detected as promo
        if summary and "?" in summary:
            issues.append({
                "promo_id": p["id"],
                "issue": "question_as_promo",
                "summary": summary[:100],
                "brand": brand,
            })
    
    return issues


def health_check(db_path):
    """Basic health check on the bot."""
    if not db_path.exists():
        return {"status": "❌ DB not found"}
    
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    health = {}
    
    # Message count (last 24h)
    cur.execute("""
        SELECT COUNT(*) FROM messages 
        WHERE timestamp >= datetime('now', '-24 hours')
    """)
    health["messages_24h"] = cur.fetchone()[0]
    
    # Promo count (last 24h)
    cur.execute("""
        SELECT COUNT(*) FROM promos 
        WHERE created_at >= datetime('now', '-24 hours')
    """)
    health["promos_24h"] = cur.fetchone()[0]
    
    # Pending alerts
    cur.execute("SELECT COUNT(*) FROM pending_alerts")
    health["pending_alerts"] = cur.fetchone()[0]
    
    # Pending confirmations
    cur.execute("SELECT COUNT(*) FROM pending_confirmations")
    health["pending_confirmations"] = cur.fetchone()[0]
    
    # Recent failures
    cur.execute("""
        SELECT COUNT(*) FROM failures 
        WHERE created_at >= datetime('now', '-24 hours')
    """)
    health["failures_24h"] = cur.fetchone()[0]
    
    # Brand distribution (top 10)
    cur.execute("""
        SELECT brand, COUNT(*) as cnt FROM promos 
        WHERE created_at >= datetime('now', '-7 days')
        GROUP BY brand ORDER BY cnt DESC LIMIT 10
    """)
    health["top_brands_7d"] = cur.fetchall()
    
    # Last promo time
    cur.execute("SELECT created_at FROM promos ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    health["last_promo"] = row[0] if row else "None"
    
    # AI corrections (last 24h)
    cur.execute("""
        SELECT COUNT(*) FROM ai_corrections 
        WHERE created_at >= datetime('now', '-24 hours')
    """)
    health["corrections_24h"] = cur.fetchone()[0]
    
    # DB size
    cur.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
    size_bytes = cur.fetchone()[0]
    health["db_size_mb"] = round(size_bytes / 1024 / 1024, 1)
    
    conn.close()
    return health


async def main():
    parser = argparse.ArgumentParser(description="TanyaDFBot Maestro Dashboard")
    parser.add_argument("--hours", type=int, default=24, help="Analysis window in hours")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print(f"═══ TanyaDFBot Maestro Dashboard ═══")
    print(f"Window: last {args.hours}h | {datetime.now(WIB).strftime('%Y-%m-%d %H:%M WIB')}\n")

    # 1. Health Check
    print("▸ HEALTH CHECK")
    health = health_check(DB_PATH)
    print(f"  Messages (24h):  {health.get('messages_24h', '?')}")
    print(f"  Promos (24h):    {health.get('promos_24h', '?')}")
    print(f"  Pending alerts:  {health.get('pending_alerts', '?')}")
    print(f"  Failures (24h):  {health.get('failures_24h', '?')}")
    print(f"  Corrections:     {health.get('corrections_24h', '?')}")
    print(f"  Last promo:      {health.get('last_promo', '?')}")
    print(f"  DB size:         {health.get('db_size_mb', '?')} MB")
    if health.get("top_brands_7d"):
        brands = ", ".join(f"{b[0]}({b[1]})" for b in health["top_brands_7d"][:5])
        print(f"  Top brands 7d:   {brands}")
    print()

    # 2. Read group messages
    print("▸ READING GROUP MESSAGES...")
    try:
        group_msgs = await read_group_messages(args.hours)
        print(f"  Total messages:  {len(group_msgs)}")
    except Exception as e:
        print(f"  ❌ Failed to read group: {e}")
        group_msgs = []

    # 3. Get DB data
    print("▸ QUERYING DATABASE...")
    db_promos = get_db_promos(args.hours)
    db_msgs = get_db_messages(args.hours)
    print(f"  DB promos:       {len(db_promos)}")
    print(f"  DB messages:     {len(db_msgs)}")

    # 4. Gap Analysis
    if group_msgs:
        print("\n▸ GAP ANALYSIS — Scanning for missed promos...")
        potential = detect_potential_promos(group_msgs)
        gaps = gap_analysis(potential, db_promos)
        print(f"  Promo signals in group: {gaps['group_promo_signals']}")
        print(f"  Promos in DB:           {gaps['db_promos']}")
        print(f"  Potentially missed:     {gaps['missed_count']}")
        if gaps["missed"]:
            print(f"\n  ⚠️  TOP MISSED PROMOS:")
            for m in gaps["missed"][:10]:
                date_wib = m["date"].astimezone(WIB).strftime("%H:%M")
                print(f"  [{date_wib}] {m['sender']}: {m['text'][:120]}")
                print(f"           Signals: {', '.join(m['signals'][:5])}")
                print(f"           Link: https://t.me/{GROUP_ENTITY}/{m['id']}")
    else:
        print("\n▸ GAP ANALYSIS — Skipped (no group data)")
        gaps = {"group_promo_signals": 0, "db_promos": len(db_promos), "missed_count": 0, "missed": []}

    # 5. Quality Audit
    if db_promos:
        print(f"\n▸ QUALITY AUDIT — {len(db_promos)} promos checked...")
        issues = quality_audit(db_promos)
        if issues:
            print(f"  ⚠️  Issues found: {len(issues)}")
            by_type = {}
            for i in issues:
                by_type[i["issue"]] = by_type.get(i["issue"], 0) + 1
            for issue_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"    {issue_type}: {count}")
            print()
            for i in issues[:10]:
                print(f"  [{i['issue']}] promo#{i['promo_id']}: {i.get('summary', '')[:80]}")
                if "mentioned" in i:
                    print(f"    → Brand: {i['brand']}, but mentions: {i['mentioned']}")
                if "status" in i:
                    print(f"    → Status: {i['status']}, raw hint: {i.get('raw_hint', '')[:60]}")
        else:
            print(f"  ✅ All promos look clean!")
    else:
        print("\n▸ QUALITY AUDIT — No promos to audit")
        issues = []

    # Summary
    print(f"\n═══ SUMMARY ═══")
    print(f"  Health:     {'✅ OK' if health.get('failures_24h', 0) == 0 else '⚠️ ' + str(health.get('failures_24h')) + ' failures'}")
    print(f"  Coverage:   {len(db_promos)}/{len(db_promos) + gaps['missed_count']} detected ({round(len(db_promos) / max(len(db_promos) + gaps['missed_count'], 1) * 100)}%)")
    print(f"  Quality:    {'✅ Clean' if not issues else f'⚠️ {len(issues)} issues'}")
    print(f"═══════════════════════════════════════")

    if args.json:
        result = {
            "health": health,
            "gap_analysis": gaps,
            "quality_issues": issues,
            "group_messages": len(group_msgs),
        }
        print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
