# CryptanAI: Classical Cipher Lab

(AI assisted Tool): AI-powered cryptanalysis of classical ciphers: automatic **detection** (which
of 15 classical ciphers produced this ciphertext?) and **cracking** (recover
the key and the plaintext) using real frequency analysis, Kasiski/Friedman
key-length recovery, hill-climbing search, and a trained ML classifier — not
a lookup table. Also encrypts, the other direction, if you just want to play
with the ciphers directly.

A study aid and a fun applied-ML project, not a security tool. See
[`PROJECT_QA.md`](PROJECT_QA.md) for the full reasoning on why classical
ciphers are still a genuinely good target for ML/NLP despite being
cryptographically obsolete, and for an extension roadmap if you want to take
this further.

```
"WKSH XEB KIAAB AOMFTHYW TWHQKXWO"  ->  Vigenere, key "LEMON"
                                     ->  "MEET ME AT THE OLD BRIDGE"
```

## What's here

- **15 classical ciphers**, each in its own independent module: no cipher
  file imports another, so adding or debugging one can never corrupt
  another. Caesar, Atbash, Affine, Substitution, Keyword, Polybius Square
  (monoalphabetic) · Vigenere, Beaufort, Autokey, Gronsfeld (polyalphabetic)
  · Hill, Playfair (polygraphic) · Rail Fence, Columnar Transposition,
  Scytale (transposition).
- **Real solvers**, not brute-force-everything: frequency analysis,
  Kasiski examination + an IC-scan (Friedman-test-style) for key length,
  quadgram-guided hill-climbing/simulated annealing for large key spaces,
  and closed-form linear algebra for Hill's known-plaintext attack.
- **A trained classifier** (scikit-learn RandomForest) that proposes which
  cipher likely produced a piece of ciphertext, feeding a pipeline that
  actually attempts to solve the top candidates and re-ranks by real
  decryption quality — see [Accuracy](#accuracy-and-honesty-about-it) below
  for why that two-stage design matters.
- **A Streamlit UI** with three tabs: paste, upload (.txt/.md/.docx/.pdf —
  formatting is stripped down to plain prose automatically), or type
  ciphertext and get the top 3 cracked candidates, each with a "Why this
  ranking?" plain-language explanation and the classifier's confidence as
  a percentage; pick a cipher and encrypt a message; or drop in any text
  and see its raw cryptanalytic statistics (index of coincidence,
  chi-squared, letter-frequency chart vs. expected English) directly —
  useful on its own, independent of solving.
- **127 passing tests**, an English-language model built from ~2.6M words
  of real (public-domain) text, and honest, tested documentation of what
  does and doesn't work reliably — see [Solver reliability](#solver-reliability-by-cipher)
  below.

## Architecture

```
Cryptanalysis/
├── ciphers/            # 15 independent cipher modules + registry.py (the
│                        # one file allowed to know about all of them)
├── solvers/             # attack strategies -- imports ciphers/, never the
│                        # reverse. One module per cipher or closely related
│                        # family (vigenere/beaufort/gronsfeld share one,
│                        # since they share one attack)
├── ml/                  # nlp_scorer (English-ness scoring), features,
│                        # dataset_generator, cipher_detector (the trained
│                        # classifier), and the bundled language-model data
├── utils/               # shared, cipher-agnostic primitives: text cleaning,
│                        # index of coincidence, chi-squared, n-gram stats
├── app/
│   ├── pipeline.py       # detect_and_solve(), explain_result(), analyze_text():
│   │                      # classifier + solvers tied together, plus the
│   │                      # explainability/analysis layer -- pure functions
│   ├── file_extraction.py # txt/md/docx/pdf -> plain text, for file upload
│   └── ui.py             # the Streamlit app itself (thin: layout only)
├── tests/                # pytest suite (127 tests)
├── scripts/
│   ├── build_language_model.py  # (dev-only) regenerates ml/corpus/*.json
│   │                              # from NLTK's public-domain Gutenberg
│   │                              # sample
│   └── verify_roundtrip.py       # standalone cipher round-trip check
├── models/cipher_detector.joblib # the trained classifier, shipped directly
├── main.py               # quick CLI smoke test across all 15 ciphers
└── PROJECT_QA.md          # design reasoning, cipher history, extension ideas
```

**Dependency direction is one-way and enforced by convention:**
`utils` ← `ciphers` ← `solvers` ← `ml` ← `app`. Nothing downstream is
imported by something upstream.

## Installation

```bash
git clone <this-repo>
cd Cryptanalysis
uv pip install -e .
```

That's the whole install for using the app — the trained classifier and
language-model statistics ship as data files (`models/`, `ml/corpus/`), so
nothing needs to be trained or downloaded first.

Regenerating the language model or retraining the classifier needs one
extra, dev-only dependency (NLTK, for public-domain training text):

```bash
uv pip install -e ".[dev]"
```

## Usage

```bash
# Quick sanity check: round-trips all 15 ciphers with random keys
python main.py

# The actual app
streamlit run app/ui.py

# Run the test suite
pytest

# Retrain the classifier from scratch (only needed if you want to; a
# trained model is already included)
python -m ml.cipher_detector
```

## The 15 ciphers, and what solving them actually looks like

Frequency analysis is the throughline: every solver here is some form of
"guess a key, measure how English the result looks, keep the best guess" —
what differs is how the *guessing* has to work, because each cipher's
mathematical structure defeats naive frequency analysis differently:

| Family | Ciphers | How the key is found |
|---|---|---|
| Monoalphabetic | Caesar, Affine | Brute force (26 / 312 keys) — small enough to try every one exactly. |
| | Substitution, Keyword | 26! possible keys — too many to brute force. Hill-climbing guided by a quadgram language model (the standard technique for this problem). Keyword cipher's ciphertext is statistically identical to plain substitution's, so one solver covers both. |
| | Polybius | Format detection (all-digit ciphertext), not frequency analysis — the unkeyed grid decodes directly. |
| Polyalphabetic | Vigenere, Beaufort, Gronsfeld | Kasiski examination + an IC-scan (the idea behind the classical Friedman test) find the key *length*; per-column chi-squared frequency analysis then finds each key *letter* — genuine two-stage frequency analysis, not a shortcut. |
| | Autokey | The keystream is nearly message-length, so the above doesn't apply — only the short primer is attackable, via a length-and-content search. Best-effort by design. |
| Polygraphic | Hill | **Deliberately not ciphertext-only.** A block cipher's matrix structure leaves no single-symbol statistical residue for frequency analysis to grab onto — the real attack is a known-plaintext "crib": given a guessed plaintext fragment, the key falls out of `K = C·P⁻¹ (mod 26)`, exact linear algebra. Both a known-key and a known-plaintext-crib tool are in the app. |
| | Playfair | Grid-position hill-climbing (same idea as Substitution, over a 5x5 grid instead of a flat alphabet). Best-effort — see below. |
| Transposition | Rail Fence, Scytale | Small integer key (rail/row count) — brute force. |
| | Columnar | Column order search: exhaustive for keywords up to 7 letters (exact), hill-climbing beyond that (best-effort). |

## Accuracy, and honesty about it

The classifier's **exact**-cipher accuracy is a modest 50.7% (15-way
classification, held-out test set). That's not a shortcoming to round up —
it's a real, tested finding with a real cause: **Caesar, Affine,
Substitution, and Keyword are ciphertext-only indistinguishable from each
other** (all relabel letters with no other observable trace of which
specific rule was used), and the same is true within
**Vigenere/Beaufort/Gronsfeld**. No amount of better modeling recovers
information that genuinely isn't in the ciphertext.

What the classifier is actually good at is *narrowing the field*:

| Metric | Accuracy |
|---|---|
| Exact cipher (15-way) | 50.7% |
| **Cipher family** (4-way: monoalphabetic / polyalphabetic / polygraphic / transposition) | **94.3%** |
| True cipher somewhere in the top 3 | **88.9%** |
| True cipher somewhere in the top 5 | **99.6%** |

That's exactly why the app doesn't stop at the classifier's raw top-1 guess:
`app/pipeline.py` takes its top-5 candidates, actually **runs the matching
solver** for each one, and re-ranks by how English the *solved* plaintext
really is — grounded in whether decryption worked, not just a classification
probability. Two engineered features do a lot of the classifier's work:
digit-ratio (Polybius's all-digit ciphertext is trivially separable — 100%
accuracy) and token-length variance (Playfair's output is *always* grouped
in exact 2-character pairs — also ~100%).

## Solver reliability by cipher

Tested directly during development, not estimated:

- **Exact, high confidence:** Caesar, Affine, Atbash, Substitution, Keyword,
  Vigenere, Beaufort, Gronsfeld, Rail Fence, Scytale, Columnar (≤7-letter
  keyword), Hill (given a valid known-plaintext crib).
- **Best-effort, honestly labeled as such in the UI:** Autokey (primer
  search can fail on longer primers), Playfair (grid hill-climbing on a 25!
  search space is a genuinely harder landscape — verified directly: the
  true grid scores clearly higher than anything a tested 80,000-move search
  budget found, and switching to simulated annealing didn't close the gap
  either; this matches Playfair's real cryptographic history of resisting
  casual cryptanalysis far longer than simple substitution did), Columnar
  with a >7-letter keyword (479 million possible column orders for a
  12-column keyword is a lot to search with a simple coordinate-ascent
  swap).
- **Not attempted by design:** blind (ciphertext-only) Hill cracking — see
  the table above for why that's a real mathematical boundary, not a gap;
  keyed Polybius grids (only the standard unkeyed grid auto-decodes).

## Data provenance

`ml/corpus/*.json` (English letter/bigram/quadgram frequency statistics) and
`ml/corpus/english_words.txt` (dictionary for word-validity scoring) are
computed from NLTK's bundled Gutenberg sample (18 public-domain books —
Austen, Melville, Milton, the King James Bible, and others, ~2.6M words) and
NLTK's `words` corpus. `scripts/build_language_model.py` documents and
reproduces exactly how; nothing at runtime needs NLTK or a network
connection — the computed statistics are what's actually shipped and used.

