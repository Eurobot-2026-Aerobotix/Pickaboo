from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ROOT_DIR = Path(__file__).resolve().parent
PREPARED_DATA_FILE = ROOT_DIR / "clean_data" / "prepared_data.json"
MODEL_FILE = ROOT_DIR / "pickaboo_v2.h5"
SCALER_FILE = ROOT_DIR / "scaler.npy"


def load_prepared_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load prepared data and extract features and labels."""
    with path.open("r", encoding="utf-8") as handle:
        batches = json.load(handle)

    X = []
    y = []

    for batch in batches:
        batch_features = []
        cam_distance = batch["cam_distance"]

        # Process each member
        for member in batch["members"]:
            member_features = [
                member["x"],
                member["y"],
                member["z"],
                member["deviation"],
                1.0 if member["blue"] else 0.0,
                cam_distance,
            ]
            batch_features.extend(member_features)

        X.append(batch_features)
        y.append(1 if batch["pickable"] else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_model() -> keras.Model:
    """Build the neural network model with deviation features linked to last hidden layer."""
    input_layer = keras.Input(shape=(24,))  # 4 members * 6 features

    # Lambda layers to split input into 4 parts
    part1 = layers.Lambda(lambda x: x[:, 0:6])(input_layer)
    part2 = layers.Lambda(lambda x: x[:, 6:12])(input_layer)
    part3 = layers.Lambda(lambda x: x[:, 12:18])(input_layer)
    part4 = layers.Lambda(lambda x: x[:, 18:24])(input_layer)

    # Create 4 parallel branches, each with 4 nodes
    branch1 = layers.Dense(4, activation="relu")(part1)
    branch2 = layers.Dense(4, activation="relu")(part2)
    branch3 = layers.Dense(4, activation="relu")(part3)
    branch4 = layers.Dense(4, activation="relu")(part4)

    # Concatenate all branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Extract deviation features (index 3, 9, 15, 21 from input)
    deviation1 = layers.Lambda(lambda x: x[:, 3:4])(input_layer)
    deviation2 = layers.Lambda(lambda x: x[:, 9:10])(input_layer)
    deviation3 = layers.Lambda(lambda x: x[:, 15:16])(input_layer)
    deviation4 = layers.Lambda(lambda x: x[:, 21:22])(input_layer)

    # Concatenate all deviations
    all_deviations = layers.Concatenate()([deviation1, deviation2, deviation3, deviation4])

    # Two fully connected layers with 12 nodes each
    fc1 = layers.Dense(12, activation="relu")(concatenated)
    fc2 = layers.Dense(12, activation="relu")(fc1)

    # Link deviations to last hidden layer by concatenating them
    fc2_with_deviations = layers.Concatenate()([fc2, all_deviations])

    # Output layer for binary classification
    output = layers.Dense(1, activation="sigmoid")(fc2_with_deviations)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model


def main() -> None:
    if not PREPARED_DATA_FILE.exists():
        raise FileNotFoundError(f"Prepared data file not found: {PREPARED_DATA_FILE}")

    print("Loading data...")
    X, y = load_prepared_data(PREPARED_DATA_FILE)
    print(f"Loaded {len(X)} batches")

    # Randomize data
    print("Randomizing data...")
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split into train (80%), validation (10%), test (10%)
    print("Splitting data into train/val/test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    print("Building model...")
    model = build_model()
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val_scaled, y_val),
        verbose=1,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model and scaler
    print(f"\nSaving model to {MODEL_FILE}")
    model.save(MODEL_FILE)

    print(f"Saving scaler to {SCALER_FILE}")
    np.save(SCALER_FILE, scaler)


if __name__ == "__main__":
    main()
