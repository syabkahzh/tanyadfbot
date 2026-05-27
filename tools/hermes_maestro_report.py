from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_reports import build_maestro_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Hermes maestro report for command center, review, and tuning.")
    parser.add_argument("--db-path", default=None, help="Override the sqlite database path.")
    parser.add_argument("--command-hours", type=int, default=2, help="Lookback window for the command center view.")
    parser.add_argument("--review-hours", type=int, default=24, help="Lookback window for review and tuning views.")
    parser.add_argument("--log-path", default=None, help="Optional runtime log file to include in the command center view.")
    parser.add_argument("--tail-lines", type=int, default=20, help="How many log lines to include when a log path is supplied.")
    args = parser.parse_args()

    print(
        build_maestro_report(
            db_path=args.db_path,
            command_hours=args.command_hours,
            review_hours=args.review_hours,
            log_path=args.log_path,
            tail_lines=args.tail_lines,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
