# Observed Prototype Predictions

These outputs were observed during local testing using the pre-trained ReviewLens AI model artifact (`model/reviewlens_model.pkl`).

---

## 1. Positive Prediction Example

- **Sentence**: `the laptop is very good`
- **Aspect**: `laptop`
- **Predicted Sentiment**: POSITIVE
- **Reported Confidence**: 84.00%

### Class Probabilities
- **Positive**: 84.00%
- **Negative**: 8.46%
- **Neutral**: 4.45%
- **Conflict**: 3.09%

---

## 2. Misclassified Negative Example (Keyboard)

- **Sentence**: `The keyboard feels cheap and uncomfortable.`
- **Aspect**: `keyboard`
- **Predicted Sentiment**: POSITIVE (Incorrect)
- **Expected Sentiment**: Negative
- **Reported Confidence**: 42.33%

### Class Probabilities
- **Positive**: 42.33%
- **Negative**: 32.63%
- **Neutral**: 18.18%
- **Conflict**: 6.86%

---

## 3. Misclassified Negative Example (Battery Life)

- **Sentence**: `The battery life is terrible and disappointing.`
- **Aspect**: `battery life`
- **Predicted Sentiment**: POSITIVE (Incorrect)
- **Expected Sentiment**: Negative
- **Reported Confidence**: 50.89%

### Class Probabilities
- **Positive**: 50.89%
- **Negative**: 29.18%
- **Neutral**: 11.95%
- **Conflict**: 7.98%

---

## Analysis & Limitations

1. **Obvious Sentiment Errors**: The linear TF-IDF model struggles with certain strongly negative terms and complex phrasing when combined with aspect markers, erroneously tilting toward the positive majority class.
2. **Uncalibrated Confidence**: Reported probabilities reflect softmax outputs of a regularized linear model and should not be interpreted as true calibrated certainty.
3. **Class Imbalance**: The training distribution has significant imbalance (particularly with very few conflict examples), which affects classification across classes.
