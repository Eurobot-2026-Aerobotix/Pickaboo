from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT_DIR = Path(__file__).resolve().parent
TEST_SAMPLE_FILE = ROOT_DIR / "test_sample.json"
FX_POSES_FILE = ROOT_DIR / "fx_poses.json"
SCALER_FILE = ROOT_DIR / "scaler.npy"
MODEL_FILES = [
    ROOT_DIR / "pickaboo.h5",
    ROOT_DIR / "pickaboo_augmented.h5",
    ROOT_DIR / "pickaboo_exagerated.h5",
    ROOT_DIR / "pickaboo_v2.h5",
    ROOT_DIR / "pickaboo_exagerated_v2.h5",
]

BLUE_BY_ARUCO_ID = {
    36: True,
    47: False,
}


def round3(value: float) -> float:
    return round(float(value), 3)


def normalize_deviation(delta: float) -> float:
    while delta > 90:
        delta -= 180
    while delta < -90:
        delta += 180
    return delta


def load_fx_poses(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        poses = json.load(handle)

    result: dict[str, dict[str, float]] = {}
    for pose in poses:
        pose_id = pose["id"]
        result[pose_id] = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "phi": float(pose["phi"]),
        }
    return result


def prepare_batch(sample: dict, fx_poses: dict[str, dict[str, float]]) -> dict:
    known_id = sample["known_id"]
    known_pose = fx_poses[known_id]

    prepared_members = []
    for member in sample["members"]:
        aruco_id = int(member["aruco_id"])
        if aruco_id not in BLUE_BY_ARUCO_ID:
            raise ValueError(f"Unsupported aruco_id in sample: {aruco_id}")

        member_phi = round3(member["phi"])
        deviation = normalize_deviation(member_phi - known_pose["phi"])
        prepared_members.append(
            {
                "x": round3(member["x"] - known_pose["x"]),
                "y": round3(member["y"] - known_pose["y"]),
                "z": round3(member["z"]),
                "deviation": round3(deviation),
                "blue": BLUE_BY_ARUCO_ID[aruco_id],
            }
        )

    cam_distance = math.hypot(known_pose["x"], known_pose["y"] - 1.5)
    return {
        "batch_id": sample.get("batch_id"),
        "members": prepared_members,
        "cam_distance": round3(cam_distance),
    }


def build_feature_vector(prepared_batch: dict) -> np.ndarray:
    features = []
    cam_distance = prepared_batch["cam_distance"]
    for member in prepared_batch["members"]:
        features.extend(
            [
                member["x"],
                member["y"],
                member["z"],
                member["deviation"],
                1.0 if member["blue"] else 0.0,
                cam_distance,
            ]
        )
    return np.array([features], dtype=np.float32)


def main() -> None:
    with TEST_SAMPLE_FILE.open("r", encoding="utf-8") as handle:
        sample = json.load(handle)

    fx_poses = load_fx_poses(FX_POSES_FILE)
    prepared = prepare_batch(sample, fx_poses)
    X = build_feature_vector(prepared)

    scaler = np.load(SCALER_FILE, allow_pickle=True).item()
    X_scaled = scaler.transform(X)

    print("Prepared batch:")
    print(json.dumps(prepared, indent=2))
    print()

    found_model = False
    for model_file in MODEL_FILES:
        if not model_file.exists():
            continue
        found_model = True
        model = tf.keras.models.load_model(model_file, compile=False, safe_mode=False)
        probability = float(model.predict(X_scaled, verbose=0)[0][0])
        predicted_pickable = probability >= 0.5

        print(f"Model: {model_file.name}")
        print(f"  probability_pickable: {probability:.6f}")
        print(f"  predicted_label: {'pickable' if predicted_pickable else 'non-pickable'}")
        print()

    if not found_model:
        raise FileNotFoundError("No model files found in MODEL_FILES")


if __name__ == "__main__":
    main()
