"""Train a Random Forest to imitate Pac-Man player actions."""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


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
        "--trees",
        type=int,
        default=300,
        help="Number of trees in the forest.",
    )
    return parser


def main():
    args = build_parser().parse_args()

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

    model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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
    joblib.dump(model, args.model_out)
    print(f"\nSaved model to {args.model_out}")


if __name__ == "__main__":
    main()
