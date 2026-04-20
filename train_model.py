"""Train a Random Forest to imitate Pac-Man player actions."""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def default_scaler_path(model_out: str) -> str:
    model_path = Path(model_out)
    project_root = Path(__file__).resolve().parent

    if not model_path.is_absolute():
        model_path = project_root / model_path

    scaler_dir = project_root / "scaler"
    return str(scaler_dir / f"{model_path.stem}_scaler.joblib")


def scale_frame(scaler: StandardScaler, frame: pd.DataFrame) -> pd.DataFrame:
    scaled = scaler.transform(frame)
    return pd.DataFrame(scaled, columns=frame.columns, index=frame.index)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/training_data.csv",
        help="Path to the logged gameplay CSV.",
    )
    parser.add_argument(
        "--model-out",
        default="models/pacman_random_forest.joblib",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--scaler-out",
        help="Where to save the fitted StandardScaler. Defaults to scaler/<model-name>_scaler.joblib.",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=300,
        help="Number of trees in the forest.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    scaler_out = args.scaler_out or default_scaler_path(args.model_out)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Training data not found at {args.data}. Play the game first to create it."
        )

    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError("Training CSV is empty. Collect gameplay data first.")

    if "label" not in df.columns:
        raise ValueError("Training CSV must contain a 'label' column.")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scale_frame(scaler, X_train)
    X_test_scaled = scale_frame(scaler, X_test)

    model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    print(f"Training rows: {len(df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    print("\nTop feature importances:")
    for name, value in importances.items():
        print(f"  {name}: {value:.4f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_out), exist_ok=True)
    joblib.dump(model, args.model_out)
    joblib.dump(scaler, scaler_out)
    print(f"\nSaved model to {args.model_out}")
    print(f"Saved scaler to {scaler_out}")


if __name__ == "__main__":
    main()
