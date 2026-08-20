"""
Round-trip correctness tests for every registered cipher: for many
random keys and varied texts (including edge cases), decrypt(encrypt(x))
should recover x -- modulo each cipher's own documented, deliberate
transformations (case-folding, J/I merging, letters-only stripping),
which are handled explicitly below rather than papered over.
"""
import pytest

from ciphers.registry import list_ciphers
from ciphers.playfair import _prepare_digraphs

TEST_TEXTS = [
    "HELLO FROM THE CRYPTO PROJECT!",
    "The Quick Brown Fox Jumps Over The Lazy Dog.",
    "A",
    "AB",
    "SHORT",
    "This is a considerably longer sentence with punctuation and Mixed CASE.",
]

CIPHER_SPECS = list_ciphers()


def _expected_for(spec, text: str) -> str:
    """What decrypt(encrypt(text)) should equal for this cipher family,
    accounting for each cipher's own documented transformations."""
    if spec.slug == "playfair":
        return "".join(_prepare_digraphs(text))
    if spec.family == "transposition":
        return text  # case-preserving, by design (see ciphers/railfence.py)
    return text.upper()  # every other family folds to uppercase, by design


@pytest.mark.parametrize("spec", CIPHER_SPECS, ids=[s.slug for s in CIPHER_SPECS])
@pytest.mark.parametrize("text", TEST_TEXTS)
def test_round_trip(spec, text):
    if spec.slug == "polybius" and any(c.isdigit() for c in text):
        pytest.skip("literal digits in plaintext are a documented, inherent "
                    "limitation of Polybius's all-digit ciphertext alphabet")

    for _ in range(3):  # a few random keys per (cipher, text) pair
        key = spec.random_key()
        ciphertext = spec.encrypt(text, key)
        plaintext = spec.decrypt(ciphertext, key)

        if spec.slug == "polybius":
            assert plaintext == text.upper().replace("J", "I")
        elif spec.slug == "hill":
            clean_original = "".join(c for c in text.upper() if c.isalpha())
            clean_dec = "".join(c for c in plaintext.upper() if c.isalpha())
            assert clean_dec[:len(clean_original)] == clean_original
        else:
            assert plaintext == _expected_for(spec, text)


def test_registry_has_fifteen_ciphers():
    assert len(CIPHER_SPECS) == 15


def test_every_cipher_has_a_working_random_key():
    for spec in CIPHER_SPECS:
        key = spec.random_key()
        # Should not raise, and should produce a non-empty ciphertext.
        assert spec.encrypt("TEST MESSAGE", key) != ""


def test_families_are_internally_consistent():
    from ciphers.registry import list_families

    families = list_families()
    total = sum(len(specs) for specs in families.values())
    assert total == 15
    for family_name, specs in families.items():
        assert all(s.family == family_name for s in specs)
