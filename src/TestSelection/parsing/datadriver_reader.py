from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_datadriver_csv(
    csv_path: Path | str,
    template_name: str,
    delimiter: str = ";",
) -> list[dict[str, Any]]:
    """Read a DataDriver CSV and return test case dicts.

    The CSV follows DataDriver convention:
    - First column header is '*** Test Cases ***'
    - Subsequent columns are variable names (e.g., ${username})
    - Rows starting with # are comments and are skipped

    Args:
        csv_path: Path to the DataDriver CSV file.
        template_name: Name of the template keyword in the .robot file.
        delimiter: CSV delimiter (DataDriver defaults to ';').

    Returns:
        List of dicts with name, description, source, is_datadriver keys.
    """
    csv_path = Path(csv_path)
    tests: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            test_name = row.get("*** Test Cases ***", "").strip()
            if not test_name or test_name.startswith("#"):
                continue
            args = {
                k: v
                for k, v in row.items()
                if k is not None and k.startswith("${") and v
            }
            description = f"Template: {template_name}. Test: {test_name}."
            if args:
                description += " " + " ".join(
                    f"{k}={v}" for k, v in args.items()
                )
            tests.append(
                {
                    "name": test_name,
                    "description": description,
                    "source": str(csv_path),
                    "is_datadriver": True,
                }
            )
    return tests
