import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

DATA_PATH = "data/Laptop_Train_v2.csv"
MODEL_PATH = "model/reviewlens_model.pkl"


def mark_aspect(sentence, aspect):
    """
    Mark the first occurrence of the aspect term in the sentence.
    Matches the preprocessing used during inference.
    """
    sentence = str(sentence)
    aspect = str(aspect)
    pattern = re.compile(re.escape(aspect), re.IGNORECASE)
    match = pattern.search(sentence)

    if match:
        start, end = match.span()
        return (
            sentence[:start]
            + "[ASPECT] "
            + sentence[start:end]
            + " [/ASPECT]"
            + sentence[end:]
        )

    # Fallback if aspect is not found in the sentence
    return f"aspect: {aspect} sentence: {sentence}"


def train():
    # 1. Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please provide the dataset before training."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Total records: {len(df)}")

    # 2. Prepare aspect-aware input
    df["text"] = df.apply(
        lambda row: mark_aspect(row["Sentence"], row["Aspect Term"]),
        axis=1,
    )

    X = df["text"]
    y = df["polarity"]

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # 4. Build ReviewLens model
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ])

    # 5. Train
    print("\nTraining ReviewLens AI...")
    model.fit(X_train, y_train)
    print("Training completed.")

    # 6. Evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n========== REVIEWLENS RESULTS ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train()
