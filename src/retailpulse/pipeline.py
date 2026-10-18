from __future__ import annotations

import argparse
from pathlib import Path

from retailpulse.data import load_retail_data


def process_file(source: Path, destination: Path) -> int:
    """Normalize one source file and save the validated rows as CSV."""
    data = load_retail_data(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a retail CSV/XLSX file for RetailPulse.")
    parser.add_argument("source", type=Path, help="Path to a raw CSV/XLSX file")
    parser.add_argument("destination", type=Path, help="Path for the normalized CSV output")
    arguments = parser.parse_args()
    row_count = process_file(arguments.source, arguments.destination)
    print(f"Wrote {row_count:,} normalized rows to {arguments.destination}")


if __name__ == "__main__":
    main()