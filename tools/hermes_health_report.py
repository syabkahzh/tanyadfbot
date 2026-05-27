from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_reports import build_service_health_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Hermes-ready service health report.")
    parser.add_argument("--db-path", default=None, help="Override the sqlite database path.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours.")
    parser.add_argument("--log-path", default=None, help="Optional service log file to tail.")
    parser.add_argument("--tail-lines", type=int, default=20, help="How many log lines to include.")
    args = parser.parse_args()

    print(
        build_service_health_report(
            db_path=args.db_path,
            hours=args.hours,
            log_path=args.log_path,
            tail_lines=args.tail_lines,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
