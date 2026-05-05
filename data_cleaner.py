from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR / "raw_data"
CLEAN_DATA_DIR = ROOT_DIR / "clean_data"
OUTPUT_FILE = CLEAN_DATA_DIR / "all_data.json"


def is_pickable(path: Path) -> bool:
	return not path.parent.name.endswith("_np")


def load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def build_track_phi_map(tracks: list[dict[str, Any]]) -> dict[int, float]:
	phi_map: dict[int, float] = {}
	for track in tracks:
		track_id = track.get("track_id")
		phi = track.get("map_yaw_deg")
		if isinstance(track_id, int) and isinstance(phi, (int, float)):
			phi_map[track_id] = float(phi)
	return phi_map


def clean_batch(batch: dict[str, Any], phi_map: dict[int, float], pickable: bool) -> dict[str, Any] | None:
	members = batch.get("members")
	if not isinstance(members, list) or len(members) != 4:
		return None

	cleaned_members: list[dict[str, Any]] = []
	for member in members:
		if not isinstance(member, dict):
			return None

		track_id = member.get("track_id")
		aruco_id = member.get("aruco_id")
		map_pos = member.get("map_pos")

		if not isinstance(track_id, int):
			return None
		if not isinstance(map_pos, list) or len(map_pos) != 3:
			return None
		if track_id not in phi_map:
			return None

		cleaned_members.append(
			{
				"aruco_id": aruco_id,
				"x": map_pos[0],
				"y": map_pos[1],
				"z": map_pos[2],
				"phi": phi_map[track_id],
			}
		)

	return {
		"batch_id": batch.get("batch_id"),
		"members": cleaned_members,
		"average_distance": batch.get("avg_adjacent_dist_m"),
		"known_id": batch.get("known_id"),
		"pickable": pickable,
	}


def clean_file(path: Path) -> list[dict[str, Any]]:
	data = load_json(path)
	tracks = data.get("tracks", [])
	batches = data.get("batches", [])

	if not isinstance(tracks, list) or not isinstance(batches, list):
		return []

	phi_map = build_track_phi_map(tracks)
	pickable = is_pickable(path)

	cleaned_batches: list[dict[str, Any]] = []
	for batch in batches:
		if not isinstance(batch, dict):
			continue

		cleaned_batch = clean_batch(batch, phi_map, pickable)
		if cleaned_batch is not None:
			cleaned_batches.append(cleaned_batch)

	return cleaned_batches


def main() -> None:
	if not RAW_DATA_DIR.exists():
		raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

	all_batches: list[dict[str, Any]] = []
	for json_path in sorted(RAW_DATA_DIR.rglob("*.json")):
		if json_path.name == "session_meta.json":
			continue
		all_batches.extend(clean_file(json_path))

	CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
	with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
		json.dump(all_batches, handle, indent=2)
		handle.write("\n")

	print(f"Wrote {len(all_batches)} batches to {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
