"""
Generates labeled synthetic training data for the cipher classifier:
samples varied real-English excerpts, encrypts each with a randomly
chosen cipher and key from the registry, and extracts features -- giving
(features, label) pairs to train ml/cipher_detector.py on.

This is a development-time tool, not a runtime dependency of the shipped
project: it needs NLTK (for varied source text) to REGENERATE training
data, but the trained model itself is shipped as a plain joblib file
(models/cipher_detector.joblib) that end users load directly -- the same
"compute once, ship the result" pattern scripts/build_language_model.py
uses for the language statistics.

Usage:
    pip install nltk
    python -m ml.cipher_detector    # regenerates data AND retrains
"""
import random

from ciphers.registry import list_ciphers
from ml.features import extract_features

MIN_EXCERPT_LEN = 150
MAX_EXCERPT_LEN = 400


def _load_source_texts() -> list[str]:
    """Loads varied real-English source text for generating training
    excerpts. Requires NLTK -- see module docstring."""
    import nltk

    nltk.download("gutenberg", quiet=True)
    from nltk.corpus import gutenberg

    return [gutenberg.raw(fid) for fid in gutenberg.fileids()]


def _random_excerpt(source_texts: list[str]) -> str:
    text = random.choice(source_texts)
    length = random.randint(MIN_EXCERPT_LEN, MAX_EXCERPT_LEN)
    start = random.randint(0, max(1, len(text) - length))
    excerpt = text[start:start + length]
    # Collapse whitespace/newlines to single spaces -- matches the kind
    # of input a user would actually paste in, rather than raw book
    # formatting with line breaks.
    return " ".join(excerpt.split())


def generate_dataset(examples_per_cipher: int = 300, seed: int = 42):
    """Returns (X, y): X is a list of feature vectors, y is the matching
    list of cipher slugs. Produces up to examples_per_cipher * 15 total
    examples (occasionally fewer, if some random key/excerpt
    combinations are skipped)."""
    random.seed(seed)
    source_texts = _load_source_texts()

    X, y = [], []
    for spec in list_ciphers():
        generated = 0
        attempts = 0
        while generated < examples_per_cipher and attempts < examples_per_cipher * 3:
            attempts += 1
            excerpt = _random_excerpt(source_texts)
            key = spec.random_key()
            try:
                ciphertext = spec.encrypt(excerpt, key)
            except Exception:
                continue  # rare invalid key/excerpt combos -- skip and retry
            if len(ciphertext.strip()) < MIN_EXCERPT_LEN // 2:
                continue
            X.append(extract_features(ciphertext))
            y.append(spec.slug)
            generated += 1

    return X, y


if __name__ == "__main__":
    X, y = generate_dataset()
    print(f"Generated {len(X)} examples across {len(set(y))} cipher types.")
