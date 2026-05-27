from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_reports import build_alert_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Hermes-ready alert quality report.")
    parser.add_argument("--db-path", default=None, help="Override the sqlite database path.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours.")
    args = parser.parse_args()

    print(build_alert_quality_report(db_path=args.db_path, hours=args.hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
