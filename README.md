# ReviewLens AI — Aspect-Based Sentiment Analysis

ReviewLens AI is a machine-learning prototype designed to predict sentiment toward a specified aspect within a laptop review sentence. It was developed as a model-integration prototype for the SWYNEX AI Internship (Task 2).

---

## Architecture & Preprocessing

The model uses a scikit-learn pipeline composed of:
1. **Aspect-Aware Preprocessing**: The target aspect is marked directly within the sentence using position tokens: `... [ASPECT] {aspect} [/ASPECT] ...`. If the exact aspect is not found, it falls back to prefix formatting `aspect: {aspect} sentence: {sentence}`.
2. **Feature Extraction**: `TfidfVectorizer` with lowercase normalization, unigram and bigram ranges (`ngram_range=(1, 2)`), sublinear term frequency scaling (`sublinear_tf=True`), and a vocabulary cap of 10,000 features.
3. **Classification**: `LogisticRegression` with `class_weight="balanced"` and `max_iter=2000` to predict one of four sentiment classes: `positive`, `negative`, `neutral`, or `conflict`.

---

## Evaluation Results

The aspect-marked Kaggle model was evaluated on a stratified held-out test split of 472 samples:

| Metric | Result |
|---|---:|
| Test Samples | 472 |
| Overall Accuracy | 71.82% |
| Macro F1-Score | 55.87% |

### Class Support & Distribution
- **Conflict**: 9 samples (severely underrepresented)
- **Negative**: 173 samples
- **Neutral**: 92 samples
- **Positive**: 198 samples

> **Note on Performance**: The conflict class has very few training and test instances. The model also misclassifies some clearly negative sentences as positive. Reported probability outputs are model estimates and must not be treated as calibrated certainty. See [`examples/predictions.md`](examples/predictions.md) for detailed observations.

---

## Local Setup & Inference

Requires Python 3.9 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run interactive prediction CLI
python predict.py
```

The pre-trained model artifact is located at `model/reviewlens_model.pkl`.

---

## Retraining & Dataset Provenance

To retrain the model:
1. Obtain the SemEval-2014 Task 4 Laptop dataset from an authorized source.
2. Place the CSV file at `data/Laptop_Train_v2.csv`.
3. Run `python train.py`.

```bash
python train.py
```

> **Dataset Provenance Caveat**: The raw dataset CSV is excluded from version control via `.gitignore` because redistribution permissions have not been verified. Retraining will overwrite `model/reviewlens_model.pkl`.

---

## Project Structure

```
swynex/
├── .gitignore               # Git exclusions (.venv, data/*.csv, caches, etc.)
├── README.md                # Project documentation and architecture
├── requirements.txt         # Pinned Python package dependencies
├── predict.py               # Interactive CLI for aspect-based sentiment inference
├── train.py                 # Training & evaluation pipeline (manual execution)
├── model/
│   └── reviewlens_model.pkl # Kaggle-trained model artifact
├── examples/
│   └── predictions.md       # Observed predictions and failure analysis
└── data/                    # Dataset directory (raw CSV excluded from Git)
```

---

## Limitations

- **Educational Prototype**: This is a baseline prototype for internship demonstration, not a production-grade service.
- **Vocabulary & Domain Sensitivity**: Performance is specific to laptop reviews and domain vocabulary.
- **No Credentials Required**: Operates completely offline with local scikit-learn models; no external APIs or secrets are used.

---

## References

- SemEval-2014 Task 4: Aspect Based Sentiment Analysis: https://alt.qcri.org/semeval2014/task4/
- Pontiki et al., *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*: https://aclanthology.org/S14-2004/
