from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
INPUT_FILE = ROOT_DIR / "clean_data" / "exagerated_data.json"
OUTPUT_FILE = ROOT_DIR / "clean_data" / "exagerated_absolute_data.json"


def absolute_deviation_batch(batch: dict[str, Any]) -> dict[str, Any]:
    transformed = copy.deepcopy(batch)

    members = transformed.get("members", [])
    if not isinstance(members, list):
        return transformed

    for member in members:
        if not isinstance(member, dict):
            continue
        if "deviation" in member:
            member["deviation"] = round(abs(float(member["deviation"])), 3)

    return transformed


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        batches = json.load(handle)

    if not isinstance(batches, list):
        raise ValueError("exagerated_data.json must contain a list of batches")

    output_batches: list[dict[str, Any]] = []
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        output_batches.append(absolute_deviation_batch(batch))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(output_batches, handle, indent=2)
        handle.write("\n")

    print(f"Input batches: {len(batches)}")
    print(f"Output batches: {len(output_batches)}")
    print(f"Wrote absolute-deviation dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
