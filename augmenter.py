from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator


ROOT_DIR = Path(__file__).resolve().parent
INPUT_FILE = ROOT_DIR / "clean_data" / "prepared_data.json"
OUTPUT_FILE = ROOT_DIR / "clean_data" / "augmented_data.json"
DEFAULT_MAX_ROWS = 5_000_000

X_Y_OFFSETS = [round(i / 1000, 3) for i in range(-4, 5, 2)]
DEVIATION_OFFSETS = [round(i * 0.8, 1) for i in range(-5, 6)]


def load_batches(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("prepared_data.json must contain a list of batches")

    return data


def normalize_deviation(value: float) -> float:
    while value > 90:
        value -= 180
    while value < -90:
        value += 180
    return round(value, 3)


def apply_offset(member: dict[str, Any], x_offset: float, y_offset: float, deviation_offset: float) -> dict[str, Any]:
    return {
        "x": round(float(member["x"]) + x_offset, 3),
        "y": round(float(member["y"]) + y_offset, 3),
        "z": round(float(member["z"]), 3),
        "deviation": normalize_deviation(float(member["deviation"]) + deviation_offset),
        "blue": bool(member["blue"]),
    }


def member_variants(member: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for x_offset in X_Y_OFFSETS:
        for y_offset in X_Y_OFFSETS:
            for deviation_offset in DEVIATION_OFFSETS:
                if x_offset == 0 and y_offset == 0 and deviation_offset == 0:
                    continue
                yield apply_offset(member, x_offset, y_offset, deviation_offset)


def augment_batch(batch: dict[str, Any], changed_member_counts: tuple[int, ...]) -> Iterator[dict[str, Any]]:
    members = batch.get("members")
    if not isinstance(members, list) or len(members) != 4:
        return

    yield {
        "batch_id": batch.get("batch_id"),
        "members": deepcopy(members),
        "pickable": bool(batch["pickable"]),
        "cam_distance": round(float(batch["cam_distance"]), 3),
    }

    member_indices = range(len(members))
    for changed_count in sorted({count for count in changed_member_counts if 1 <= count <= len(members)}):
        for chosen_indices in itertools.combinations(member_indices, changed_count):
            variant_lists = [list(member_variants(members[index])) for index in chosen_indices]
            for chosen_variants in itertools.product(*variant_lists):
                augmented_members = deepcopy(members)
                for index, variant in zip(chosen_indices, chosen_variants):
                    augmented_members[index] = variant

                yield {
                    "batch_id": batch.get("batch_id"),
                    "members": augmented_members,
                    "pickable": bool(batch["pickable"]),
                    "cam_distance": round(float(batch["cam_distance"]), 3),
                }


def write_augmented_data(
    input_path: Path,
    output_path: Path,
    changed_member_counts: tuple[int, ...],
    limit_batches: int | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    dry_run: bool = False,
) -> int:
    batches = load_batches(input_path)
    if limit_batches is not None:
        batches = batches[:limit_batches]

    total_rows = 0
    if dry_run:
        for batch in batches:
            for _ in augment_batch(batch, changed_member_counts):
                total_rows += 1
                if total_rows >= max_rows:
                    return total_rows
        return total_rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first_item = True
        for batch_index, batch in enumerate(batches, start=1):
            print(f"Augmenting batch {batch_index}/{len(batches)}")
            for augmented_batch in augment_batch(batch, changed_member_counts):
                if not first_item:
                    handle.write(",\n")
                json.dump(augmented_batch, handle)
                first_item = False
                total_rows += 1
                if total_rows >= max_rows:
                    handle.write("\n]\n")
                    return total_rows
        handle.write("\n]\n")

    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment prepared batch data.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="Path to prepared_data.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Path to augmented_data.json")
    parser.add_argument(
        "--member-counts",
        type=int,
        nargs="+",
        default=[1],
        help="Exact member counts to perturb per augmented sample. Default is 1.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Only process the first N batches for a smaller run",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Hard cap on the number of augmented rows to generate. Default is 5,000,000.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count the augmented rows without writing a file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_rows = write_augmented_data(
        input_path=args.input,
        output_path=args.output,
        changed_member_counts=tuple(args.member_counts),
        limit_batches=args.limit_batches,
        max_rows=args.max_rows,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"Dry run: would write {total_rows} augmented rows")
    else:
        if total_rows >= args.max_rows:
            print(f"Wrote {total_rows} augmented rows to {args.output} (capped at {args.max_rows})")
        else:
            print(f"Wrote {total_rows} augmented rows to {args.output}")


if __name__ == "__main__":
    main()