"""tools/hermes_control.py — CLI for Hermes to control Tanya's runtime.

Write config overrides and send commands to the Tanya processing loop via the
hermes_config and hermes_commands DB tables.

Usage:
    PYTHONPATH=. .venv/bin/python tools/hermes_control.py show-config
    PYTHONPATH=. .venv/bin/python tools/hermes_control.py set-config <key> <value>
    PYTHONPATH=. .venv/bin/python tools/hermes_control.py delete-config <key>
    PYTHONPATH=. .venv/bin/python tools/hermes_control.py send-command <command> [payload-json]
    PYTHONPATH=. .venv/bin/python tools/hermes_control.py list-commands [--status pending|done|failed]
    PYTHONPATH=. .venv/bin/python tools/hermes_control.py list-recent [--limit 20]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import Database


async def show_config(db: Database) -> int:
    config = await db.get_all_hermes_config()
    if not config:
        print("No Hermes config set. Tanya is using defaults.")
        return 0
    print("# Hermes Runtime Config\n")
    print(f"| Key | Value |")
    print(f"|-----|-------|")
    for k, v in sorted(config.items()):
        print(f"| `{k}` | `{v}` |")
    return 0


async def set_config(db: Database, key: str, value: str) -> int:
    await db.set_hermes_config(key, value)
    print(f"Set `{key}` = `{value}`")
    return 0


async def delete_config(db: Database, key: str) -> int:
    await db.delete_hermes_config(key)
    print(f"Deleted `{key}`")
    return 0


async def send_command(db: Database, command: str, payload: str) -> int:
    # Validate JSON
    try:
        json.loads(payload)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON payload: {e}", file=sys.stderr)
        return 1
    cmd_id = await db.send_hermes_command(command, payload)
    print(f"Queued command #{cmd_id}: {command}")
    return 0


async def list_commands(db: Database, status: str | None) -> int:
    if status:
        commands = await db.get_pending_commands() if status == "pending" else await db.get_recent_commands(50)
        commands = [c for c in commands if c["status"] == status]
    else:
        commands = await db.get_recent_commands(50)
    if not commands:
        print("No commands found.")
        return 0
    print(f"| ID | Command | Status | Created |")
    print(f"|----|---------|--------|---------|")
    for c in commands:
        print(f"| {c['id']} | {c['command']} | {c['status']} | {c['created_at']} |")
    return 0


async def list_recent(db: Database, limit: int) -> int:
    commands = await db.get_recent_commands(limit)
    if not commands:
        print("No commands found.")
        return 0
    print(f"| ID | Command | Status | Result | Created |")
    print(f"|----|---------|--------|--------|---------|")
    for c in commands:
        result = (c["result"] or "")[:40]
        print(f"| {c['id']} | {c['command']} | {c['status']} | {result} | {c['created_at']} |")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes control CLI for Tanya runtime config and commands.")
    sub = parser.add_subparsers(dest="subcmd", required=True)

    sub.add_parser("show-config", help="Show all Hermes config values")

    p_set = sub.add_parser("set-config", help="Set a config value")
    p_set.add_argument("key", help="Config key")
    p_set.add_argument("value", help="Config value")

    p_del = sub.add_parser("delete-config", help="Delete a config key")
    p_del.add_argument("key", help="Config key")

    p_cmd = sub.add_parser("send-command", help="Send a command to Tanya")
    p_cmd.add_argument("command", help="Command type (reprocess, override_alert, suppress_brand, force_alert)")
    p_cmd.add_argument("payload", nargs="?", default="{}", help="JSON payload")

    p_list = sub.add_parser("list-commands", help="List commands")
    p_list.add_argument("--status", help="Filter by status (pending, done, failed)")

    p_recent = sub.add_parser("list-recent", help="List recent commands")
    p_recent.add_argument("--limit", type=int, default=20, help="Max results")

    args = parser.parse_args()

    db = Database()
    await db.init()

    try:
        if args.subcmd == "show-config":
            return await show_config(db)
        elif args.subcmd == "set-config":
            return await set_config(db, args.key, args.value)
        elif args.subcmd == "delete-config":
            return await delete_config(db, args.key)
        elif args.subcmd == "send-command":
            return await send_command(db, args.command, args.payload)
        elif args.subcmd == "list-commands":
            return await list_commands(db, args.status)
        elif args.subcmd == "list-recent":
            return await list_recent(db, args.limit)
        else:
            parser.print_help()
            return 1
    finally:
        if db.conn:
            await db.conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
