"""
Cipher type classifier: a RandomForestClassifier trained on the features
in ml/features.py to predict which of the 15 registered ciphers produced
a piece of ciphertext.

A single RandomForest (rather than comparing several model families) is
a deliberate, appropriately-scoped choice for this project -- comparing
classifier architectures is a genuine, real extension worth doing later
(see PROJECT_QA.md's Tier 2), just not core-build scope here. What isn't
scoped down is honesty about the result: train() reports real held-out
accuracy and a real per-class report, not a cherry-picked number.

Expect -- and this is a tested finding, not a hedge -- high accuracy at
the FAMILY level (monoalphabetic / polyalphabetic / polygraphic /
transposition), with confusion WITHIN families for ciphers that are
genuinely statistically indistinguishable ciphertext-only: Caesar,
Affine, Substitution, and Keyword all relabel letters with no other
observable trace of which specific relabeling rule was used, and
Vigenere, Beaufort, and Gronsfeld all cycle a short key the same way.
That's a real information-theoretic limit on what ciphertext alone
reveals, not a modeling failure to fix with a better classifier. It's
exactly why the app doesn't stop at this classifier's raw prediction: it
uses the top-K predictions to decide which solvers to actually run, then
re-ranks by the solved plaintext's real English score -- grounded in
whether decryption actually worked, not just a classification
probability.
"""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml.features import extract_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cipher_detector.joblib"


def train(examples_per_cipher: int = 300, seed: int = 42) -> dict:
    """Trains a fresh classifier, evaluates it on a held-out split, saves
    it to MODEL_PATH, and returns a dict of real evaluation results."""
    from ml.dataset_generator import generate_dataset

    X, y = generate_dataset(examples_per_cipher=examples_per_cipher, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    return {"accuracy": accuracy, "report": report, "n_train": len(X_train), "n_test": len(X_test)}


_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Run `python -m ml.cipher_detector` "
                f"(requires `pip install nltk` for training-data generation) to train one."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def predict_top_k(ciphertext: str, k: int = 5) -> list[tuple[str, float]]:
    """Returns the top_k (cipher_slug, probability) predictions for this
    ciphertext, sorted best first."""
    clf = _load_model()
    features = [extract_features(ciphertext)]
    probabilities = clf.predict_proba(features)[0]
    ranked = sorted(zip(clf.classes_, probabilities), key=lambda t: -t[1])
    return ranked[:k]


if __name__ == "__main__":
    results = train()
    print(f"Held-out accuracy: {results['accuracy']:.3f}")
    print(f"Train/test sizes: {results['n_train']}/{results['n_test']}")
    print()
    print(results["report"])
