"""
Loads the bundled English language model (ml/corpus/*.json + english_words.txt)
and exposes the statistical primitives built on top of it:

  - chi_squared_statistic(text): how far the text's letter distribution is
    from expected English (low = more English-like)
  - bigram_log_likelihood(text) / quadgram_log_likelihood(text): how
    plausible the text's letter sequences are under an English n-gram model
  - dictionary_word_ratio(text): fraction of whitespace-separated tokens
    that are real English words

These are *primitives*. ml/nlp_scorer.py combines them into the single
composite "how English is this?" score that both the solvers and the
detector use -- this module doesn't make that judgment call itself, it
just provides the individual measurements.

The data files were generated once by scripts/build_language_model.py from
NLTK's public-domain Gutenberg sample and word corpus. Nothing at runtime
needs NLTK or a network connection -- these are plain JSON/text files
shipped with the repo.
"""
import json
from functools import lru_cache
from pathlib import Path

from utils.text_utils import ALPHABET, clean_letters_only

CORPUS_DIR = Path(__file__).resolve().parent.parent / "ml" / "corpus"

# Reference English letter frequencies (the classic ETAOIN SHRDLU
# distribution), loaded from the bundled corpus stats. Used for the
# chi-squared goodness-of-fit test.
EXPECTED_LETTER_FREQ = {}
BIGRAM_LOGPROB = {}
QUADGRAM_LOGPROB = {}
QUADGRAM_FLOOR = -10.0
ENGLISH_WORDS = set()


@lru_cache(maxsize=1)
def _load():
    """
    Loads all bundled data files exactly once per process.

    Mutates the module-level dicts/set IN PLACE (.update(), not
    reassignment) rather than rebinding them to new objects. This
    matters: `from utils.language_stats import EXPECTED_LETTER_FREQ`
    elsewhere in the codebase binds a local name to whatever object
    EXPECTED_LETTER_FREQ pointed to *at import time* -- if this function
    later did `EXPECTED_LETTER_FREQ = json.load(f)` (rebinding to a new
    dict), that new object would never be visible through the earlier
    import, which would still see the original empty dict. Mutating the
    same dict object in place means every existing reference -- import
    order doesn't matter -- sees the update. (QUADGRAM_FLOOR is a bare
    float and can't be fixed this way; nothing currently imports it
    directly, and it should stay that way -- call quadgram_log_likelihood()
    or the other functions in this module instead of reaching for the
    raw globals.)
    """
    global QUADGRAM_FLOOR

    with open(CORPUS_DIR / "unigram_freq.json") as f:
        EXPECTED_LETTER_FREQ.update(json.load(f))

    with open(CORPUS_DIR / "bigram_logprob.json") as f:
        BIGRAM_LOGPROB.update(json.load(f))

    with open(CORPUS_DIR / "quadgram_logprob.json") as f:
        raw_quad = json.load(f)
        QUADGRAM_FLOOR = raw_quad.pop("__FLOOR__")
        QUADGRAM_LOGPROB.update(raw_quad)

    with open(CORPUS_DIR / "english_words.txt") as f:
        ENGLISH_WORDS.update(f.read().split())

    return True


def chi_squared_statistic(text: str) -> float:
    """
    Pearson's chi-squared statistic comparing this text's letter
    distribution against expected English letter frequencies. Lower is
    more English-like; 0 would be a perfect match.

    This is the classical, textbook tool for scoring Caesar/Affine/
    substitution candidate decryptions -- it only looks at single-letter
    frequency, which is exactly the statistic those ciphers leave intact
    (they relabel letters, they don't change how often each label is used
    relative to the others).
    """
    _load()
    letters = clean_letters_only(text)
    n = len(letters)
    if n == 0:
        return float("inf")

    observed = {ch: 0 for ch in ALPHABET}
    for ch in letters:
        observed[ch] += 1

    chi2 = 0.0
    for ch in ALPHABET:
        expected_count = EXPECTED_LETTER_FREQ[ch] * n
        if expected_count > 0:
            chi2 += (observed[ch] - expected_count) ** 2 / expected_count
    return chi2


def bigram_log_likelihood(text: str) -> float:
    """Average log10-probability of the text's letter bigrams under the
    English bigram model. Higher (closer to 0) is more English-like."""
    _load()
    letters = clean_letters_only(text)
    if len(letters) < 2:
        return QUADGRAM_FLOOR

    total = 0.0
    count = 0
    for i in range(len(letters) - 1):
        gram = letters[i:i + 2]
        total += BIGRAM_LOGPROB.get(gram, QUADGRAM_FLOOR)
        count += 1
    return total / count


def quadgram_log_likelihood(text: str) -> float:
    """
    Average log10-probability of the text's letter quadgrams under the
    English quadgram model -- the gold-standard fitness function for
    hill-climbing/simulated-annealing substitution solvers. Higher
    (closer to 0) is more English-like.

    Unseen quadgrams fall back to a smoothed floor value rather than
    -infinity, so one unlucky 4-letter window can't veto an otherwise
    excellent candidate decryption.
    """
    _load()
    letters = clean_letters_only(text)
    if len(letters) < 4:
        return QUADGRAM_FLOOR

    total = 0.0
    count = 0
    for i in range(len(letters) - 3):
        gram = letters[i:i + 4]
        total += QUADGRAM_LOGPROB.get(gram, QUADGRAM_FLOOR)
        count += 1
    return total / count


def dictionary_word_ratio(text: str) -> float:
    """
    Fraction of whitespace-separated tokens that are real English words
    (punctuation stripped from token edges before lookup). Returns 0.0 for
    text with no tokens at all.

    This is the most intuitive signal for a human reading the UI ("62% of
    words are real English words") and is particularly valuable for
    transposition ciphers, where letter/bigram statistics alone can't
    distinguish a correct unscrambling from an incorrect one -- word
    formation is the real tell.
    """
    _load()
    tokens = text.upper().split()
    if not tokens:
        return 0.0

    hits = 0
    for tok in tokens:
        word = "".join(ch for ch in tok if ch in ALPHABET)
        if word and word in ENGLISH_WORDS:
            hits += 1
    return hits / len(tokens)


def is_english_word(word: str) -> bool:
    _load()
    return word.upper() in ENGLISH_WORDS
