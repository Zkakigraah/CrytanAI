"""
Builds the bundled English-language reference data used by ml/nlp_scorer.py
and every solver in solvers/.

This script is NOT a runtime dependency of the project. It is a one-time
(re-runnable) data-provenance script: it downloads public-domain text via
NLTK, computes letter/bigram/quadgram statistics from it, and writes the
results to ml/corpus/*.json and ml/corpus/english_words.txt.

Those output files are what the project actually ships and imports at
runtime -- nobody who clones the repo needs NLTK or a network connection
to use CryptoAI. You only need to re-run this script if you want to
regenerate the language model from a different or larger corpus.

Data sources (both public domain / permissively licensed):
  - Text statistics: NLTK's bundled Gutenberg sample (18 public-domain
    books -- Austen, Melville, Milton, the King James Bible, etc.),
    ~2.6M words / ~11.8M characters.
  - Dictionary: NLTK's bundled 'words' corpus (derived from the classic
    /usr/share/dict word list tradition), ~236k entries.

Usage:
    pip install nltk
    python scripts/build_language_model.py
"""
import json
import math
import string
from collections import Counter
from pathlib import Path

ALPHABET = string.ascii_uppercase
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml" / "corpus"

# Quadgrams occurring fewer than this many times in the whole corpus are
# dropped from the shipped table -- they're statistical noise, and pruning
# them keeps the bundled JSON small. Unseen quadgrams fall back to a
# smoothed floor value at scoring time (see ml/nlp_scorer.py).
MIN_QUADGRAM_COUNT = 3


def fetch_corpus_text() -> str:
    """Downloads (if needed) and returns the raw Gutenberg sample text."""
    import nltk

    nltk.download("gutenberg", quiet=True)
    nltk.download("words", quiet=True)
    from nltk.corpus import gutenberg

    return gutenberg.raw()


def fetch_word_list() -> list[str]:
    import nltk

    nltk.download("words", quiet=True)
    from nltk.corpus import words

    return words.words()


def clean_letters_only(text: str) -> str:
    """Uppercase, alphabetic characters only -- matches how every cipher
    in this project treats text, so the statistics line up with what the
    solvers will actually see."""
    return "".join(ch for ch in text.upper() if ch in ALPHABET)


def build_ngram_tables(letters: str) -> tuple[dict, dict, dict]:
    n = len(letters)

    # --- Unigram (letter) frequency, normalized to a probability dist. ---
    unigram_counts = Counter(letters)
    total_letters = sum(unigram_counts.values())
    unigram_freq = {
        ch: unigram_counts.get(ch, 0) / total_letters for ch in ALPHABET
    }

    # --- Bigram log-probabilities (dense: 26x26 = 676 possible keys) ---
    bigram_counts = Counter(letters[i:i + 2] for i in range(n - 1))
    total_bigrams = sum(bigram_counts.values())
    # Laplace smoothing so every one of the 676 possible bigrams gets a
    # finite log-probability, even ones absent from the corpus.
    vocab_bigram = 26 * 26
    bigram_logprob = {}
    for a in ALPHABET:
        for b in ALPHABET:
            key = a + b
            count = bigram_counts.get(key, 0)
            prob = (count + 1) / (total_bigrams + vocab_bigram)
            bigram_logprob[key] = round(math.log10(prob), 6)

    # --- Quadgram log-probabilities (sparse: pruned + floor value) ---
    quad_counts = Counter(letters[i:i + 4] for i in range(n - 3))
    total_quads = sum(quad_counts.values())
    quad_logprob = {
        key: round(math.log10(count / total_quads), 6)
        for key, count in quad_counts.items()
        if count >= MIN_QUADGRAM_COUNT
    }
    # Floor: what an unseen quadgram is scored as. Standard smoothing
    # technique for quadgram-based fitness functions -- treat unseen
    # quadgrams as if they occurred ~0.01 times, rather than assigning
    # them -infinity (which would let a single bad 4-gram window veto an
    # otherwise-excellent candidate decryption).
    floor = round(math.log10(0.01 / total_quads), 6)
    quad_logprob["__FLOOR__"] = floor

    return unigram_freq, bigram_logprob, quad_logprob


def build_word_list(raw_words: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for w in raw_words:
        w = w.upper()
        if w.isalpha() and 1 < len(w) <= 20 and w not in seen:
            seen.add(w)
            cleaned.append(w)
    return sorted(cleaned)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching corpus text (NLTK Gutenberg sample)...")
    raw_text = fetch_corpus_text()
    letters = clean_letters_only(raw_text)
    print(f"  {len(letters):,} letters after cleaning.")

    print("Computing n-gram frequency tables...")
    unigram, bigram, quad = build_ngram_tables(letters)
    print(f"  unigram: {len(unigram)} entries")
    print(f"  bigram : {len(bigram)} entries")
    print(f"  quadgram: {len(quad)} entries (pruned at count >= {MIN_QUADGRAM_COUNT})")

    print("Fetching and cleaning dictionary word list...")
    raw_words = fetch_word_list()
    word_list = build_word_list(raw_words)
    print(f"  {len(word_list):,} words after cleaning/dedup.")

    (OUTPUT_DIR / "unigram_freq.json").write_text(json.dumps(unigram, indent=2))
    (OUTPUT_DIR / "bigram_logprob.json").write_text(json.dumps(bigram, indent=2))
    (OUTPUT_DIR / "quadgram_logprob.json").write_text(json.dumps(quad, indent=2))
    (OUTPUT_DIR / "english_words.txt").write_text("\n".join(word_list))

    print(f"\nWrote language model files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
