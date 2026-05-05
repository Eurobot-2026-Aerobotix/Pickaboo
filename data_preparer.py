from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
INPUT_FILE = ROOT_DIR / "clean_data" / "all_data.json"
FX_POSES_FILE = ROOT_DIR / "fx_poses.json"
OUTPUT_FILE = ROOT_DIR / "clean_data" / "prepared_data.json"

BLUE_BY_ARUCO_ID = {
    36: True,
    47: False,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def round3(value: Any) -> float:
    return round(float(value), 3)


def normalize_deviation(delta: float) -> float:
    while delta > 90:
        delta -= 180
    while delta < -90:
        delta += 180
    return delta


def load_fx_poses(path: Path) -> dict[str, dict[str, float]]:
    poses = load_json(path)
    if not isinstance(poses, list):
        raise ValueError("fx_poses.json must contain a list of pose objects")

    result: dict[str, dict[str, float]] = {}
    for pose in poses:
        if not isinstance(pose, dict):
            continue

        pose_id = pose.get("id")
        if not isinstance(pose_id, str):
            continue

        result[pose_id] = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "phi": float(pose["phi"]),
        }

    return result


def prepare_member(member: dict[str, Any], known_pose: dict[str, float]) -> dict[str, Any] | None:
    aruco_id = member.get("aruco_id")
    if aruco_id not in BLUE_BY_ARUCO_ID:
        return None

    member_phi = round3(member["phi"])
    deviation = normalize_deviation(member_phi - known_pose["phi"])

    return {
        "x": round3(member["x"] - known_pose["x"]),
        "y": round3(member["y"] - known_pose["y"]),
        "z": round3(member["z"]),
        "deviation": round3(deviation),
        "blue": BLUE_BY_ARUCO_ID[aruco_id],
    }


def prepare_batch(batch: dict[str, Any], fx_poses: dict[str, dict[str, float]]) -> dict[str, Any] | None:
    known_id = batch.get("known_id")
    if not isinstance(known_id, str):
        return None

    known_pose = fx_poses.get(known_id)
    if known_pose is None:
        return None

    members = batch.get("members")
    if not isinstance(members, list):
        return None

    prepared_members = []
    for member in members:
        if not isinstance(member, dict):
            return None
        prepared_member = prepare_member(member, known_pose)
        if prepared_member is None:
            return None
        prepared_members.append(prepared_member)

    cam_distance = math.hypot(known_pose["x"], known_pose["y"] - 1.5)

    return {
        "batch_id": batch.get("batch_id"),
        "members": prepared_members,
        "pickable": bool(batch["pickable"]),
        "cam_distance": round3(cam_distance),
    }


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    if not FX_POSES_FILE.exists():
        raise FileNotFoundError(f"Pose file not found: {FX_POSES_FILE}")

    batches = load_json(INPUT_FILE)
    if not isinstance(batches, list):
        raise ValueError("all_data.json must contain a list of batches")

    fx_poses = load_fx_poses(FX_POSES_FILE)
    prepared_batches = []
    skipped_batches = 0
    for batch in batches:
        prepared_batch = prepare_batch(batch, fx_poses)
        if prepared_batch is None:
            skipped_batches += 1
            continue
        prepared_batches.append(prepared_batch)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(prepared_batches, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {len(prepared_batches)} batches to {OUTPUT_FILE}")
    if skipped_batches:
        print(f"Skipped {skipped_batches} batches without usable pose data")


if __name__ == "__main__":
    main()
