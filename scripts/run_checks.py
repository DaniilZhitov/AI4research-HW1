"""Run the eight correctness checks and write results/checks.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moire_dirac.checks import CHECKS  # noqa: E402


def main() -> int:
    output = ROOT / "results" / "checks.csv"
    output.parent.mkdir(exist_ok=True)
    results = [check() for check in CHECKS]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["check", "tolerance", "max_deviation", "passed",
             "control_deviation", "control_threshold"]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    f"{result.tolerance:.0e}",
                    f"{result.max_deviation:.3e}",
                    result.passed,
                    "" if result.control_deviation is None else f"{result.control_deviation:.3e}",
                    "" if result.control_threshold is None else f"{result.control_threshold:.0e}",
                ]
            )
            print(result)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
