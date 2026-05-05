from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


ROOT_DIR = Path(__file__).resolve().parent
PREPARED_DATA_FILE = ROOT_DIR / "clean_data" / "prepared_data.json"
MODEL_FILE = ROOT_DIR / "pickaboo.h5"
SCALER_FILE = ROOT_DIR / "scaler.npy"
REPORT_JSON_FILE = ROOT_DIR / "statistics_report.json"
REPORT_MD_FILE = ROOT_DIR / "statistics_report.md"

FEATURE_NAMES = [
    "member1_x",
    "member1_y",
    "member1_z",
    "member1_deviation",
    "member1_blue",
    "member1_cam_distance",
    "member2_x",
    "member2_y",
    "member2_z",
    "member2_deviation",
    "member2_blue",
    "member2_cam_distance",
    "member3_x",
    "member3_y",
    "member3_z",
    "member3_deviation",
    "member3_blue",
    "member3_cam_distance",
    "member4_x",
    "member4_y",
    "member4_z",
    "member4_deviation",
    "member4_blue",
    "member4_cam_distance",
]

MEMBER_GROUPS = {
    "member1": list(range(0, 6)),
    "member2": list(range(6, 12)),
    "member3": list(range(12, 18)),
    "member4": list(range(18, 24)),
}

ATTRIBUTE_GROUPS = {
    "x": [0, 6, 12, 18],
    "y": [1, 7, 13, 19],
    "z": [2, 8, 14, 20],
    "deviation": [3, 9, 15, 21],
    "blue": [4, 10, 16, 22],
    "cam_distance": [5, 11, 17, 23],
}


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_prepared_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        batches = json.load(handle)

    features: list[list[float]] = []
    labels: list[int] = []

    for batch in batches:
        batch_features: list[float] = []
        cam_distance = float(batch["cam_distance"])

        for member in batch["members"]:
            batch_features.extend(
                [
                    float(member["x"]),
                    float(member["y"]),
                    float(member["z"]),
                    float(member["deviation"]),
                    1.0 if member["blue"] else 0.0,
                    cam_distance,
                ]
            )

        features.append(batch_features)
        labels.append(1 if batch["pickable"] else 0)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def load_scaler(path: Path) -> StandardScaler:
    loaded = np.load(path, allow_pickle=True).item()
    if not isinstance(loaded, StandardScaler):
        raise TypeError("scaler.npy does not contain a StandardScaler instance")
    return loaded


def split_data(X: np.ndarray, y: np.ndarray) -> SplitData:
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int32)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def permutation_importance(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
) -> list[dict[str, Any]]:
    baseline_probs = model.predict(X_test, verbose=0).reshape(-1)
    baseline_accuracy = accuracy_score(y_test, baseline_probs >= 0.5)

    importances: list[dict[str, Any]] = []

    for feature_index, feature_name in enumerate(feature_names):
        drops: list[float] = []
        for repeat_index in range(n_repeats):
            rng = np.random.default_rng(10_000 + feature_index * 100 + repeat_index)
            permuted = X_test.copy()
            permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
            permuted_probs = model.predict(permuted, verbose=0).reshape(-1)
            permuted_accuracy = accuracy_score(y_test, permuted_probs >= 0.5)
            drops.append(baseline_accuracy - permuted_accuracy)

        importances.append(
            {
                "feature": feature_name,
                "baseline_accuracy": float(baseline_accuracy),
                "mean_accuracy_drop": float(np.mean(drops)),
                "std_accuracy_drop": float(np.std(drops)),
            }
        )

    importances.sort(key=lambda item: item["mean_accuracy_drop"], reverse=True)
    return importances


def grouped_importance(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups: dict[str, list[int]],
    n_repeats: int = 5,
) -> list[dict[str, Any]]:
    baseline_probs = model.predict(X_test, verbose=0).reshape(-1)
    baseline_accuracy = accuracy_score(y_test, baseline_probs >= 0.5)

    results: list[dict[str, Any]] = []

    for group_name, indices in groups.items():
        drops: list[float] = []
        for repeat_index in range(n_repeats):
            rng = np.random.default_rng(20_000 + repeat_index + sum(indices))
            permuted = X_test.copy()
            shuffled_rows = rng.permutation(len(permuted))
            permuted[:, indices] = permuted[shuffled_rows][:, indices]
            permuted_probs = model.predict(permuted, verbose=0).reshape(-1)
            permuted_accuracy = accuracy_score(y_test, permuted_probs >= 0.5)
            drops.append(baseline_accuracy - permuted_accuracy)

        results.append(
            {
                "group": group_name,
                "baseline_accuracy": float(baseline_accuracy),
                "mean_accuracy_drop": float(np.mean(drops)),
                "std_accuracy_drop": float(np.std(drops)),
            }
        )

    results.sort(key=lambda item: item["mean_accuracy_drop"], reverse=True)
    return results


def write_markdown_report(report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Pickaboo Model Statistics")
    lines.append("")
    lines.append("## Test Metrics")
    metrics = report["test_metrics"]
    lines.append(f"- Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"- Precision: {metrics['precision']:.4f}")
    lines.append(f"- Recall: {metrics['recall']:.4f}")
    roc_auc = metrics["roc_auc"]
    lines.append(f"- ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "- ROC AUC: n/a")
    lines.append(f"- Confusion Matrix: {metrics['confusion_matrix']}")
    lines.append("")
    lines.append("## Most Important Individual Features")
    for item in report["feature_importance"][:10]:
        lines.append(
            f"- {item['feature']}: mean accuracy drop {item['mean_accuracy_drop']:.4f}"
        )
    lines.append("")
    lines.append("## Most Important Groups")
    lines.append("### Members")
    for item in report["member_importance"]:
        lines.append(
            f"- {item['group']}: mean accuracy drop {item['mean_accuracy_drop']:.4f}"
        )
    lines.append("### Attribute Types")
    for item in report["attribute_importance"]:
        lines.append(
            f"- {item['group']}: mean accuracy drop {item['mean_accuracy_drop']:.4f}"
        )

    REPORT_MD_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not PREPARED_DATA_FILE.exists():
        raise FileNotFoundError(f"Prepared data file not found: {PREPARED_DATA_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not SCALER_FILE.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_FILE}")

    print("Loading prepared data...")
    X, y = load_prepared_data(PREPARED_DATA_FILE)

    print("Splitting data...")
    split = split_data(X, y)

    print("Loading scaler and model...")
    scaler = load_scaler(SCALER_FILE)
    model = tf.keras.models.load_model(MODEL_FILE, safe_mode=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    X_test_scaled = scaler.transform(split.X_test)
    test_probs = model.predict(X_test_scaled, verbose=0).reshape(-1)

    print("Computing test metrics...")
    test_metrics = evaluate_predictions(split.y_test, test_probs)

    print("Computing feature importance...")
    feature_importance = permutation_importance(
        model, X_test_scaled, split.y_test, FEATURE_NAMES
    )

    print("Computing group importance...")
    member_importance = grouped_importance(
        model, X_test_scaled, split.y_test, MEMBER_GROUPS
    )
    attribute_importance = grouped_importance(
        model, X_test_scaled, split.y_test, ATTRIBUTE_GROUPS
    )

    report = {
        "dataset_size": int(len(X)),
        "split_sizes": {
            "train": int(len(split.X_train)),
            "validation": int(len(split.X_val)),
            "test": int(len(split.X_test)),
        },
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "member_importance": member_importance,
        "attribute_importance": attribute_importance,
    }

    REPORT_JSON_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(report)

    print(f"Wrote {REPORT_JSON_FILE}")
    print(f"Wrote {REPORT_MD_FILE}")
    print("Top 5 features:")
    for item in feature_importance[:5]:
        print(f"- {item['feature']}: {item['mean_accuracy_drop']:.4f}")


if __name__ == "__main__":
    main()