"""
Standalone verification script (not part of the pytest suite -- that
comes later): round-trips every registered cipher many times with random
keys and varied text, including edge cases. Run directly:

    python3 scripts/verify_roundtrip.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ciphers.registry import list_ciphers
from ciphers.playfair import _prepare_digraphs as _playfair_prepare

TEST_TEXTS = [
    "HELLO FROM THE CRYPTO PROJECT! 123",
    "The Quick Brown Fox Jumps Over The Lazy Dog.",
    "A",
    "AB",
    "SHORT",
    "This is a considerably longer sentence with punctuation, numbers (42), and Mixed CASE to stress-test every cipher's edge handling.",
]

FAILURES = []
TOTAL = 0

for spec in list_ciphers():
    for trial in range(10):
        key = spec.random_key()
        for text in TEST_TEXTS:
            TOTAL += 1
            try:
                enc = spec.encrypt(text, key)
                dec = spec.decrypt(enc, key)
            except Exception as e:
                FAILURES.append((spec.slug, key, text, f"EXCEPTION: {e}"))
                continue

            if spec.slug == "polybius" and any(c.isdigit() for c in text):
                # Documented, inherent limitation (see ciphers/polybius.py):
                # an all-digit ciphertext alphabet can't coexist with
                # literal digits in the plaintext. Not this cipher's
                # supported domain -- skip rather than mis-test it.
                continue
            elif spec.slug == "polybius":
                # J and I share a grid cell by design; decrypt always
                # recovers I. Expected, documented, not a bug.
                ok = dec == text.upper().replace("J", "I")
            elif spec.slug == "hill":
                # Hill pads with trailing 'X's appended after all original
                # content to reach a multiple of the block size; the
                # letters-only prefix must match exactly.
                clean_original = "".join(c for c in text.upper() if c.isalpha())
                clean_dec = "".join(c for c in dec.upper() if c.isalpha())
                ok = clean_dec[:len(clean_original)] == clean_original
            elif spec.slug == "playfair":
                # Playfair's own digraph-preparation (letters only, J->I,
                # fillers for double letters and odd length) tells us
                # exactly what decrypt(encrypt(text)) must equal.
                expected = "".join(_playfair_prepare(text))
                ok = dec == expected
            elif spec.family == "transposition":
                # Pure reordering ciphers preserve case exactly -- they
                # never fold to uppercase, by design (see railfence.py).
                ok = dec == text
            else:
                # Every monoalphabetic/polyalphabetic cipher uppercases
                # its input as a first step (pre-existing, documented
                # behavior) -- compare against the uppercased original.
                ok = dec == text.upper()

            if not ok:
                FAILURES.append((spec.slug, key, text, f"got: {dec!r}"))

print(f"Ran {TOTAL} round-trip checks across {len(list_ciphers())} ciphers.")
if FAILURES:
    print(f"\n{len(FAILURES)} FAILURES:")
    for slug, key, text, info in FAILURES[:30]:
        print(f"  [{slug}] key={key!r} text={text!r} -> {info}")
    sys.exit(1)
else:
    print("All round-trips passed. ✅")
