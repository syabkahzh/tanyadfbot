"""Generate a near-live Hermes shadow watch report."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_reports import build_shadow_watch_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a near-live Hermes shadow watch report.")
    parser.add_argument("--db-path", help="SQLite DB path. Defaults to Tanya config/local fallback.")
    parser.add_argument("--minutes", type=int, default=5, help="Recent shadow-watch window.")
    parser.add_argument("--quiet-empty", action="store_true", help="Print nothing when there are no findings.")
    args = parser.parse_args()
    print(
        build_shadow_watch_report(
            db_path=args.db_path,
            minutes=args.minutes,
            quiet_empty=args.quiet_empty,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
