from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
RAW_TEST_DIR = ROOT_DIR / "raw_data" / "test"
FX_POSES_FILE = ROOT_DIR / "fx_poses.json"
OUT_FILE = ROOT_DIR / "test_samples.json"
ALLOWED_ARUCO_IDS = {36, 47}


def round3(value: float) -> float:
    return round(float(value), 3)


def load_fx_poses(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        poses = json.load(handle)

    result: dict[str, dict[str, float]] = {}
    for pose in poses:
        pid = pose["id"]
        result[pid] = {"x": float(pose["x"]), "y": float(pose["y"]), "phi": float(pose.get("phi", 0))}
    return result


def nearest_pose_id(x: float, y: float, fx_poses: dict[str, dict[str, float]]) -> str:
    best_pose_id = ""
    best_distance = None
    for pose_id, pose in fx_poses.items():
        distance = math.hypot(x - pose["x"], y - pose["y"])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_pose_id = pose_id
    return best_pose_id


def build_member(track: dict[str, Any]) -> dict[str, Any] | None:
    aruco_id = int(track.get("aruco_id", -1))
    if aruco_id not in ALLOWED_ARUCO_IDS:
        return None

    position = track.get("map_pos")
    if not isinstance(position, list) or len(position) != 3:
        return None

    return {
        "track_id": int(track.get("track_id", -1)),
        "aruco_id": aruco_id,
        "x": round3(position[0]),
        "y": round3(position[1]),
        "z": round3(position[2]),
        "phi": round3(track.get("map_yaw_deg", 0.0)),
    }


def process_raw_file(path: Path, fx_poses: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    tracks = raw.get("tracks", [])
    if not isinstance(tracks, list):
        return []

    grouped_members: dict[str, list[dict[str, Any]]] = {}
    for track in tracks:
        if not isinstance(track, dict):
            continue

        member = build_member(track)
        if member is None:
            continue

        pose_id = nearest_pose_id(member["x"], member["y"], fx_poses)
        if not pose_id:
            continue

        grouped_members.setdefault(pose_id, []).append(member)

    samples: list[dict[str, Any]] = []
    base_batch_id = raw.get("_write_timestamp") or raw.get("timestamp") or path.stem

    for pose_id in sorted(grouped_members):
        members = grouped_members[pose_id]
        if len(members) != 4:
            continue

        members.sort(key=lambda item: item["track_id"])
        known_pose = fx_poses[pose_id]
        average_distance = sum(
            math.hypot(member["x"] - known_pose["x"], member["y"] - known_pose["y"])
            for member in members
        ) / len(members)

        samples.append(
            {
                "batch_id": f"{base_batch_id}_{pose_id}",
                "members": [
                    {
                        "aruco_id": member["aruco_id"],
                        "x": member["x"],
                        "y": member["y"],
                        "z": member["z"],
                        "phi": member["phi"],
                    }
                    for member in members
                ],
                "average_distance": round3(average_distance),
                "known_id": pose_id,
            }
        )

    return samples


def main() -> None:
    fx_poses = load_fx_poses(FX_POSES_FILE)

    files = sorted(RAW_TEST_DIR.glob("*.json"))
    output = []
    for f in files:
        output.extend(process_raw_file(f, fx_poses))

    if not output:
        print("No samples produced from raw_data/test")
        return

    with OUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Wrote {len(output)} sample(s) to {OUT_FILE.name}")


if __name__ == "__main__":
    main()
