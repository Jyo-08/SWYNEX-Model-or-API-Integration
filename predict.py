import joblib
import re

MODEL_PATH = "model/reviewlens_model.pkl"

# Load trained ReviewLens model
model = joblib.load(MODEL_PATH)


def mark_aspect(sentence, aspect):
    """
    Mark the first occurrence of the aspect in the sentence.
    This matches the preprocessing used during model training.
    """
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


def predict_sentiment(sentence, aspect):
    """Predict sentiment and return class probabilities."""
    model_text = mark_aspect(sentence, aspect)

    prediction = model.predict([model_text])[0]
    probabilities = model.predict_proba([model_text])[0]
    classes = model.classes_

    probability_dict = {
        class_name: probability
        for class_name, probability in zip(classes, probabilities)
    }

    confidence = max(probabilities)
    return prediction, confidence, probability_dict


def main():
    print("=" * 55)
    print("              ReviewLens AI")
    print("        Aspect-Based Sentiment Analyzer")
    print("=" * 55)

    sentence = input("\nEnter review sentence: ")
    aspect = input("Enter aspect term: ")

    prediction, confidence, probabilities = predict_sentiment(
        sentence,
        aspect,
    )

    print("\n===== Prediction =====")
    print(f"Sentence   : {sentence}")
    print(f"Aspect     : {aspect}")
    print(f"Sentiment  : {prediction.upper()}")
    print(f"Confidence : {confidence * 100:.2f}%")

    print("\n===== Class Probabilities =====")

    for sentiment in ["positive", "negative", "neutral", "conflict"]:
        probability = probabilities.get(sentiment, 0)
        print(f"{sentiment.capitalize():<10}: {probability * 100:.2f}%")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
