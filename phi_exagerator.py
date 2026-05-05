from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
INPUT_FILE = ROOT_DIR / "clean_data" / "prepared_data.json"
OUTPUT_FILE = ROOT_DIR / "clean_data" / "exagerated_data.json"


def normalize_deviation(value: float) -> float:
    while value > 90:
        value -= 180
    while value < -90:
        value += 180
    return value


def push_away_from_zero(value: float, delta: float) -> float:
    if value > 0:
        return value + delta
    if value < 0:
        return value - delta
    return delta


def exaggerate_pickable_batch(batch: dict[str, Any]) -> dict[str, Any]:
    exaggerated = copy.deepcopy(batch)

    for member in exaggerated["members"]:
        original_deviation = float(member["deviation"])
        moved_deviation = push_away_from_zero(original_deviation, 5.0)
        member["deviation"] = round(normalize_deviation(moved_deviation), 3)

    return exaggerated


def tweak_non_pickable_batch(batch: dict[str, Any]) -> dict[str, Any]:
    tweaked = copy.deepcopy(batch)

    tweaked["cam_distance"] = round(float(tweaked["cam_distance"]) * 1.01, 3)

    for member in tweaked["members"]:
        member["x"] = round(float(member["x"]) * 1.01, 3)
        member["y"] = round(float(member["y"]) * 1.01, 3)
        member["z"] = round(float(member["z"]) * 1.01, 3)
        member["deviation"] = round(
            normalize_deviation(float(member["deviation"]) * 1.01), 3
        )

    return tweaked


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        batches = json.load(handle)

    if not isinstance(batches, list):
        raise ValueError("prepared_data.json must contain a list of batches")

    output_batches: list[dict[str, Any]] = []
    pickable_count = 0
    non_pickable_count = 0

    for batch in batches:
        if not isinstance(batch, dict):
            continue

        is_pickable = bool(batch.get("pickable", False))
        output_batches.append(copy.deepcopy(batch))

        if is_pickable:
            pickable_count += 1
            output_batches.append(exaggerate_pickable_batch(batch))
        else:
            non_pickable_count += 1
            output_batches.append(tweak_non_pickable_batch(batch))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(output_batches, handle, indent=2)
        handle.write("\n")

    print(f"Input batches: {len(batches)}")
    print(f"Pickable batches: {pickable_count}")
    print(f"Non-pickable batches: {non_pickable_count}")
    print(f"Output batches: {len(output_batches)}")
    print(f"Wrote exaggerated dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
