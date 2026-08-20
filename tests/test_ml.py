"""
Tests for the ml/ package: the English scorer, feature extraction, and
the trained cipher classifier.

test_cipher_detector_* tests are skipped (not failed) if no trained model
is present at models/cipher_detector.joblib -- training requires `pip
install nltk` for training-data generation (see ml/dataset_generator.py),
which isn't assumed to be present in every environment running this
suite. The shipped repo includes a pre-trained model, so these normally
run.
"""
import math

import pytest

from ml.nlp_scorer import default_scorer
from ml.features import extract_features, FEATURE_NAMES
from ciphers.registry import get_cipher

ENGLISH_TEXT = (
    "MACHINE LEARNING HAS TRANSFORMED THE FIELD OF CRYPTANALYSIS BY "
    "ALLOWING COMPUTERS TO AUTOMATICALLY DETECT PATTERNS IN ENCRYPTED TEXT"
)
GIBBERISH_TEXT = "QXZJ KVWP ZZYX QQXV BJKQ WXZZ PPQV XKJZ MFLQ TZVX"


def test_scorer_ranks_english_above_gibberish():
    assert default_scorer.score(ENGLISH_TEXT) > default_scorer.score(GIBBERISH_TEXT)
    assert default_scorer.quick_score(ENGLISH_TEXT) > default_scorer.quick_score(GIBBERISH_TEXT)


def test_scorer_breakdown_fields_are_consistent():
    breakdown = default_scorer.full_score(ENGLISH_TEXT)
    assert breakdown.total == default_scorer.score(ENGLISH_TEXT)
    assert breakdown.word_ratio > 0.5  # most tokens in real English are real words


def test_scorer_handles_empty_and_tiny_input_without_crashing():
    for text in ["", "A", "  ", "!!!"]:
        default_scorer.score(text)  # should not raise
        default_scorer.quick_score(text)


def test_feature_vector_shape_and_finiteness():
    """Every feature must be a finite float -- sklearn can't handle NaN
    or inf, which is exactly why chi_squared is explicitly capped in
    ml/features.py (Polybius's near-empty-letter case produces inf
    otherwise)."""
    for slug in ["caesar", "polybius", "playfair", "vigenere"]:
        spec = get_cipher(slug)
        ciphertext = spec.encrypt(ENGLISH_TEXT, spec.random_key())
        features = extract_features(ciphertext)
        assert len(features) == len(FEATURE_NAMES)
        for value in features:
            assert math.isfinite(value), f"non-finite feature for {slug}: {features}"


def test_features_separate_polybius_by_digit_ratio():
    spec = get_cipher("polybius")
    ciphertext = spec.encrypt(ENGLISH_TEXT, "")
    features = dict(zip(FEATURE_NAMES, extract_features(ciphertext)))
    assert features["digit_ratio"] > 0.8


def test_features_separate_transposition_by_low_chi_squared():
    """Verified design property (see ml/features.py's docstring):
    transposition ciphers preserve letter identity, so their chi-squared
    distance from expected English frequencies stays low; monoalphabetic
    substitution relabels letters, so its chi-squared is much higher."""
    railfence_ct = get_cipher("railfence").encrypt(ENGLISH_TEXT, 4)
    substitution_ct = get_cipher("substitution").encrypt(ENGLISH_TEXT, get_cipher("substitution").random_key())

    railfence_chi2 = dict(zip(FEATURE_NAMES, extract_features(railfence_ct)))["chi_squared"]
    substitution_chi2 = dict(zip(FEATURE_NAMES, extract_features(substitution_ct)))["chi_squared"]
    assert railfence_chi2 < substitution_chi2


@pytest.fixture
def trained_model_available():
    from ml.cipher_detector import MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("No trained model present -- run `python -m ml.cipher_detector` first.")
    return True


def test_cipher_detector_predicts_plausible_top_k(trained_model_available):
    from ml.cipher_detector import predict_top_k

    spec = get_cipher("polybius")  # the cleanest, most reliably-separable case
    ciphertext = spec.encrypt(ENGLISH_TEXT, "")
    predictions = predict_top_k(ciphertext, k=3)

    assert len(predictions) == 3
    assert predictions[0][0] == "polybius"
    assert predictions[0][1] > 0.5


def test_cipher_detector_probabilities_sum_near_one(trained_model_available):
    from ml.cipher_detector import predict_top_k

    spec = get_cipher("caesar")
    ciphertext = spec.encrypt(ENGLISH_TEXT, spec.random_key())
    # Top-15 (all classes) probabilities should sum to ~1.0
    predictions = predict_top_k(ciphertext, k=15)
    total = sum(p for _, p in predictions)
    assert abs(total - 1.0) < 1e-6
