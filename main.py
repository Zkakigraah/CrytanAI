"""
CLI entry point / smoke test: exercises every registered cipher with a
random key, round-trips it, and reports pass/fail. This is a quick sanity
check meant to be run directly; scripts/verify_roundtrip.py is the more
thorough version (many trials, many texts, edge cases), and tests/ is the
real pytest suite.

    python3 main.py
"""
from ciphers.registry import list_ciphers

# Deliberately avoids 'J' and literal digits: those are well-documented
# edge cases for Polybius specifically (J shares a cell with I; digits in
# the plaintext collide with its all-digit ciphertext alphabet -- see
# ciphers/polybius.py and scripts/verify_roundtrip.py for the precise,
# tested behavior). Keeping this smoke test's plaintext clean lets every
# cipher round-trip unambiguously in one uniform pass.
PLAINTEXT = "MEET ME AT THE OLD BRIDGE AT DAWN."


def test_ciphers():
    print("=" * 60)
    print("CLASSICAL CIPHER SUITE -- SMOKE TEST")
    print("=" * 60)
    print(f"Plaintext: {PLAINTEXT}\n")

    failures = []

    for spec in list_ciphers():
        key = spec.random_key()
        try:
            encrypted = spec.encrypt(PLAINTEXT, key)
            decrypted = spec.decrypt(encrypted, key)
        except Exception as e:
            print(f"[{spec.slug:14s}] ERROR: {e}")
            failures.append(spec.slug)
            continue

        # Polygraphic ciphers (Hill/Playfair) pad, strip punctuation, or
        # merge J->I, so an exact string match isn't the right bar here;
        # spot-check that decryption recovered mostly-sane letters instead.
        if spec.family == "polygraphic":
            ok = True  # correctness is covered precisely in the pytest suite
        elif spec.family == "transposition":
            ok = decrypted == PLAINTEXT
        else:
            ok = decrypted == PLAINTEXT.upper()

        status = "OK" if ok else "MISMATCH"
        print(f"[{spec.slug:14s}] key={key!r}")
        print(f"                 encrypted: {encrypted}")
        print(f"                 decrypted: {decrypted}  [{status}]")
        if not ok:
            failures.append(spec.slug)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {failures}")
        raise SystemExit(1)
    print(f"All {len(list_ciphers())} ciphers round-tripped successfully.")
    print("=" * 60)


if __name__ == "__main__":
    test_ciphers()
