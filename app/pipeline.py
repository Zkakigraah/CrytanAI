"""
The actual "detect + solve" pipeline behind the UI's Decrypt tab.

This is the piece that makes the 90% top-3 / 99.6% top-5 classifier
accuracy (see ml/cipher_detector.py) good enough to build on, despite the
classifier's exact-cipher accuracy being much lower (~50%, for the
genuine, tested reasons documented there): rather than trusting the
classifier's single best guess, this takes its top-K candidates, actually
RUNS the matching solver for each one, and re-ranks by the solved
plaintext's real English score. A wrong-but-plausible classifier guess
gets corrected here, as long as the true cipher was anywhere in the
shortlist -- which the top-5 number says is true 99.6% of the time.

Kept separate from app/ui.py on purpose: this is business logic (pure
functions, testable without Streamlit), not UI layout.
"""
from dataclasses import dataclass

from ciphers.registry import get_cipher
from ml.cipher_detector import predict_top_k
from ml.nlp_scorer import default_scorer
from solvers.base import SolveResult
from solvers import (
    solve_caesar, solve_affine, solve_atbash, solve_substitution,
    solve_vigenere, solve_beaufort, solve_gronsfeld,
    solve_railfence, solve_scytale, solve_columnar,
    solve_autokey, solve_playfair,
)
from ciphers.polybius import decrypt_polybius

CANDIDATES_TO_ATTEMPT = 5   # cast a wider net than the final top-3 shown,
                             # to hedge against classifier imperfection --
                             # this is exactly what the top-5 (not top-3)
                             # accuracy number justifies.

MIN_PROBABILITY_TO_ATTEMPT = 0.01   # Tested finding, not a guess: a
    # solver forced to run on ciphertext it wasn't designed for still
    # returns SOME plausible-scoring nonsense (brute-force solvers in
    # particular always produce their least-bad candidate). If a
    # best-effort solver for the true cipher happens to fail that
    # attempt, a classifier-implausible wrong guess can occasionally
    # out-score it. The classifier's probabilities for genuinely
    # implausible candidates are often *exactly* zero (RandomForest: no
    # tree votes for that class at all) -- skipping those avoids the
    # failure mode at its root instead of trying to out-weight it after
    # the fact with a fragile score-blending formula.

# Not every cipher slug maps directly to a solve_X(ciphertext, top_k)
# function -- some share a solver, one has no blind solver by design.
# This is where those exceptions are handled, once, rather than forcing
# every solver into an identical shape it doesn't naturally have.
_DIRECT_SOLVERS = {
    "caesar": solve_caesar,
    "affine": solve_affine,
    "atbash": solve_atbash,
    "substitution": solve_substitution,
    "keyword": solve_substitution,   # ciphertext-indistinguishable from substitution -- see substitution_solver.py
    "vigenere": solve_vigenere,
    "beaufort": solve_beaufort,
    "gronsfeld": solve_gronsfeld,
    "autokey": solve_autokey,
    "playfair": solve_playfair,
    "railfence": solve_railfence,
    "scytale": solve_scytale,
    "columnar": solve_columnar,
}


def _attempt_polybius(ciphertext: str, top_k: int) -> list[SolveResult]:
    """Polybius has no search-based solver (see solvers/__init__.py's
    docstring on why) -- only the unkeyed standard grid can be
    auto-attempted, which needs no search at all, just a direct decode."""
    plaintext = decrypt_polybius(ciphertext, "")
    return [SolveResult(key="", plaintext=plaintext, score=default_scorer.score(plaintext),
                         method="direct_decode", confidence="medium")]


def _attempt_slug(slug: str, ciphertext: str, top_k: int) -> list[SolveResult]:
    """Runs whichever solving strategy exists for this slug. Hill is
    deliberately absent here: it has no blind ciphertext-only attack by
    design (see solvers/hill_solver.py) -- the UI surfaces this as a
    distinct case with its own known-plaintext-crib flow, not silently
    skipped."""
    if slug in _DIRECT_SOLVERS:
        return _DIRECT_SOLVERS[slug](ciphertext, top_k=top_k)
    if slug == "polybius":
        return _attempt_polybius(ciphertext, top_k)
    return []  # hill, or any future slug without an auto-solver


@dataclass
class DetectAndSolveResult:
    cipher_slug: str
    cipher_display_name: str
    cipher_family: str
    key: object
    plaintext: str
    score: float
    score_breakdown: object  # ml.nlp_scorer.ScoreBreakdown -- the "why" behind the score
    confidence: str
    method: str
    classifier_probability: float
    hill_suspected_but_needs_crib: bool = False


def detect_and_solve(ciphertext: str, top_k: int = 3) -> list[DetectAndSolveResult]:
    """
    The full pipeline: classify, solve each shortlisted candidate for
    real, and re-rank by actual decryption quality. Returns the top_k
    results, best first.
    """
    predictions = predict_top_k(ciphertext, k=CANDIDATES_TO_ATTEMPT)
    plausible = [(s, p) for s, p in predictions if p >= MIN_PROBABILITY_TO_ATTEMPT]
    # Fallback: if filtering leaves too few candidates to fill top_k
    # (only plausible on very short or unusual ciphertext), fall back to
    # the unfiltered list rather than under-delivering results.
    if len(plausible) < top_k:
        plausible = predictions

    all_results: list[DetectAndSolveResult] = []
    hill_was_candidate = False

    for slug, probability in plausible:
        if slug == "hill":
            hill_was_candidate = True
            continue

        spec = get_cipher(slug)
        try:
            solve_results = _attempt_slug(slug, ciphertext, top_k=1)
        except Exception:
            continue  # a solver failing on a mismatched cipher type is
                       # expected (e.g. trying Playfair's grid search on
                       # non-Playfair ciphertext) -- just skip it, other
                       # candidates cover the ranking.

        for r in solve_results:
            all_results.append(DetectAndSolveResult(
                cipher_slug=slug, cipher_display_name=spec.display_name, cipher_family=spec.family,
                key=r.key, plaintext=r.plaintext, score=r.score,
                score_breakdown=default_scorer.full_score(r.plaintext),
                confidence=r.confidence, method=r.method, classifier_probability=probability,
            ))

    all_results.sort(key=lambda r: -r.score)
    top_results = all_results[:top_k]

    # Hill can't be auto-solved, but if the classifier flagged it as
    # plausible, that's worth surfacing rather than silently dropping --
    # the person can then use the dedicated known-plaintext-crib flow.
    if hill_was_candidate and not any(r.cipher_slug == "hill" for r in top_results):
        hill_prob = next(p for s, p in predictions if s == "hill")
        top_results.append(DetectAndSolveResult(
            cipher_slug="hill", cipher_display_name="Hill Cipher", cipher_family="polygraphic",
            key=None, plaintext="", score=float("-inf"), score_breakdown=None,
            confidence="n/a", method="none", classifier_probability=hill_prob,
            hill_suspected_but_needs_crib=True,
        ))

    return top_results


# --------------------------------------------------------------------------
# Explainability: turns the data already computed above into a plain-
# language, step-by-step account of what happened. Deliberately built from
# fields that already exist (ScoreBreakdown, method, classifier_probability)
# rather than inventing new machinery -- this is the ScoreBreakdown
# dataclass's originally-intended use, not a new subsystem.
# --------------------------------------------------------------------------

METHOD_EXPLANATIONS = {
    "brute_force": "Tried every possible key and kept the one whose decryption scored highest — small enough key space to check exhaustively and exactly.",
    "hill_climbing": "The key space is too large to brute force, so this searched it: repeatedly tried small changes to a candidate key, kept the ones that made the decryption more English-like, and ran many independent random restarts to avoid getting stuck.",
    "kasiski+ic_scan+frequency": "Two stages: first found the key LENGTH (by checking which length makes the ciphertext split into columns that individually look like single-alphabet English), then used classical per-letter frequency analysis on each of those columns to recover the key itself.",
    "primer_search": "Autokey's key is nearly message-length, so only the short starting \"primer\" is attackable — tried every short primer exhaustively, and searched longer ones.",
    "known_key": "Decrypted directly using the key provided — no search involved.",
    "known_plaintext_crib": "Solved for the exact key using linear algebra (K = C·P⁻¹ mod 26) from the known-plaintext fragment provided — exact, not a search.",
    "direct_decode": "Polybius's standard grid has no key to search for — decoded directly.",
    "exhaustive": "Tried every possible arrangement for this key length and kept the best-scoring one — small enough to check exactly.",
    "none": "No automatic ciphertext-only attack exists for this cipher.",
}

CONFIDENCE_MEANINGS = {
    "high": "this technique reliably finds the exact key when the classifier's guess is correct.",
    "medium": "a workable signal, but with more uncertainty than the high-confidence solvers.",
    "best_effort": "this is a genuinely hard search space for this cipher — tested directly during development to sometimes not converge, even with a large search budget. Worth treating this result as a strong hint rather than a certainty, and double-checking by eye.",
    "n/a": "not applicable — no automatic attack was attempted.",
}


def explain_result(result: DetectAndSolveResult) -> list[str]:
    """Returns a short, ordered list of plain-language explanation steps
    for how this particular result was produced."""
    steps = [
        f"Classifier estimate: {result.classifier_probability:.0%} probability this is "
        f"{result.cipher_display_name} ({result.cipher_family}), based on statistical "
        f"features of the ciphertext alone (letter-frequency shape, index of coincidence, "
        f"digit/space ratios, token-length pattern, and more).",

        f"Attack attempted: {METHOD_EXPLANATIONS.get(result.method, result.method)}",
    ]

    if result.score_breakdown:
        b = result.score_breakdown
        steps.append(
            f"How English the result looks: combines letter-frequency fit "
            f"(chi-squared={b.chi_squared:.1f} — lower means closer to expected English letter "
            f"frequencies), bigram plausibility ({b.bigram_ll:.2f}), quadgram plausibility "
            f"({b.quadgram_ll:.2f} — the strongest single signal), and real-word ratio "
            f"({b.word_ratio:.0%} of tokens are dictionary words) into one score: {b.total:.3f}."
        )

    steps.append(f"Confidence: {result.confidence} — {CONFIDENCE_MEANINGS.get(result.confidence, '')}")
    return steps


# --------------------------------------------------------------------------
# Standalone statistical analysis -- independent of solving. Useful on its
# own for inspecting any text's cryptanalytic profile (the same signals the
# solvers and classifier actually use), not just after a crack attempt.
# --------------------------------------------------------------------------

@dataclass
class TextAnalysis:
    length: int
    index_of_coincidence: float
    chi_squared: float
    bigram_log_likelihood: float
    quadgram_log_likelihood: float
    word_ratio: float
    digit_ratio: float
    space_ratio: float
    avg_token_length: float
    observed_letter_freq: dict   # {letter: fraction}
    expected_letter_freq: dict   # {letter: fraction}


def analyze_text(text: str) -> TextAnalysis:
    """The exact statistics every solver and the classifier are built on,
    computed directly for inspection -- not tied to any particular cipher
    or solving attempt."""
    from collections import Counter

    from utils.text_utils import index_of_coincidence, clean_letters_only, ALPHABET
    from utils.language_stats import (
        chi_squared_statistic, bigram_log_likelihood, quadgram_log_likelihood,
        dictionary_word_ratio, EXPECTED_LETTER_FREQ, _load,
    )
    from ml.features import extract_features, FEATURE_NAMES

    _load()  # ensures EXPECTED_LETTER_FREQ is populated
    letters = clean_letters_only(text)
    counts = Counter(letters)
    total = len(letters) or 1
    observed = {ch: counts.get(ch, 0) / total for ch in ALPHABET}

    features = dict(zip(FEATURE_NAMES, extract_features(text)))

    return TextAnalysis(
        length=len(text),
        index_of_coincidence=index_of_coincidence(text),
        chi_squared=chi_squared_statistic(text),
        bigram_log_likelihood=bigram_log_likelihood(text),
        quadgram_log_likelihood=quadgram_log_likelihood(text),
        word_ratio=dictionary_word_ratio(text),
        digit_ratio=features["digit_ratio"],
        space_ratio=features["space_ratio"],
        avg_token_length=features["avg_token_length"],
        observed_letter_freq=observed,
        expected_letter_freq=dict(EXPECTED_LETTER_FREQ),
    )
