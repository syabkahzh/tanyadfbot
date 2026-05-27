from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_reports import build_recent_promo_lookup_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer recent promo lookup questions from the local Tanya database.")
    parser.add_argument("--db-path", default=None, help="Override the sqlite database path.")
    parser.add_argument("--hours", type=int, default=2, help="Lookback window in hours.")
    parser.add_argument("--today", action="store_true", help="Show promos since midnight WIB.")
    parser.add_argument("--brand", default=None, help="Optional exact brand filter.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum promos to show.")
    args = parser.parse_args()

    print(
        build_recent_promo_lookup_report(
            db_path=args.db_path,
            hours=args.hours,
            brand=args.brand,
            limit=args.limit,
            today=args.today,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
