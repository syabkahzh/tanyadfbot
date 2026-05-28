"""Generate a proactive Hermes supervisor report."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_reports import build_supervisor_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Hermes supervisor report for proactive monitoring.")
    parser.add_argument("--db-path", help="SQLite DB path. Defaults to Tanya config/local fallback.")
    parser.add_argument("--hours", type=int, default=2, help="Recent monitoring window.")
    parser.add_argument("--log-path", help="Optional Tanya runtime log path.")
    args = parser.parse_args()
    print(build_supervisor_report(db_path=args.db_path, hours=args.hours, log_path=args.log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
