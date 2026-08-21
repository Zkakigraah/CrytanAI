"""
Streamlit UI: three tabs.

  Decrypt: paste or upload ciphertext, get the top-3 most likely (cipher,
  key, decrypted text) via app/pipeline.py's detect-and-solve pipeline,
  each with a plain-language "Why?" explanation.

  Encrypt: pick a cipher and key, encrypt a message with it.

  Analyze: paste or upload any text and see its cryptanalytic statistical
  profile directly -- the same signals the solvers and classifier use,
  useful on its own for inspecting/debugging, independent of solving.

This file is UI layout only -- no cracking/classification/extraction logic
lives here, all of that is in app/pipeline.py, app/file_extraction.py,
ciphers/, solvers/, and ml/, so those pieces stay testable without
Streamlit.

Run with:
    streamlit run app/ui.py
"""
import pandas as pd
import streamlit as st

from ciphers.registry import list_families, get_cipher
from app.pipeline import detect_and_solve, explain_result, analyze_text
from app.file_extraction import extract_text, SUPPORTED_EXTENSIONS

st.set_page_config(page_title="CryptanAI · Classical Cipher Lab", page_icon="🔐", layout="centered")

SOLVER_BADGE = {
    "high":        "🟢 solver: reliable — this attack finds the exact key when "
                   "the cipher type is correct",
    "medium":      "🟡 solver: moderate — workable signal, but more uncertainty",
    "best_effort": "🟠 solver: best effort — search may not have fully converged; "
                   "treat as a strong hint and verify by eye",
    "n/a":         "⚪ solver: n/a — no automatic ciphertext-only attack exists for "
                   "this cipher (Hill needs a known-plaintext crib)",
}

FAMILY_LABELS = {
    "monoalphabetic": "Monoalphabetic (single fixed substitution)",
    "polyalphabetic": "Polyalphabetic (shifting/repeating key)",
    "polygraphic": "Polygraphic (encrypts blocks of letters at once)",
    "transposition": "Transposition (reorders letters, never relabels them)",
}


def _file_upload_text(key_prefix: str) -> str | None:
    """Shows a file uploader accepting txt/md/docx/pdf and returns the
    extracted plain text, or None if nothing's uploaded. Errors surface
    as a friendly message rather than a crash."""
    uploaded = st.file_uploader(
        "Or upload a file", type=list(SUPPORTED_EXTENSIONS), key=f"{key_prefix}_upload",
        help="Formatting (Markdown syntax, docx styling, PDF layout) is stripped down to plain "
             "prose automatically — classical ciphers only ever operated on letters.",
    )
    if uploaded is None:
        return None
    try:
        text = extract_text(uploaded.name, uploaded.getvalue())
        st.caption(f"✅ Extracted {len(text)} characters from **{uploaded.name}**.")
        return text
    except Exception as e:
        st.error(f"Couldn't read {uploaded.name}: {e}")
        return None


def _key_input_widget(spec, key_prefix: str):
    """Builds the right input widget(s) for this cipher's key shape,
    inferred from its registry key_example -- int/tuple/str/None each
    get a different, natural widget rather than forcing everything
    through a single text box."""
    example = spec.key_example

    if example is None:
        st.caption(f"🔑 {spec.key_description} — no input needed.")
        return None

    if isinstance(example, tuple):
        st.caption(f"🔑 {spec.key_description}")
        c1, c2 = st.columns(2)
        a = c1.number_input("a", min_value=1, max_value=25, value=example[0], key=f"{key_prefix}_a")
        b = c2.number_input("b", min_value=0, max_value=25, value=example[1], key=f"{key_prefix}_b")
        return (int(a), int(b))

    if isinstance(example, int):
        default = example if example else 3
        return int(st.number_input(f"🔑 {spec.key_description}", min_value=1, value=default, key=f"{key_prefix}_int"))

    return st.text_input(
        f"🔑 {spec.key_description}", value=str(example),
        placeholder=str(example) or "(leave blank for the standard grid)",
        key=f"{key_prefix}_str",
    )


def _render_result_card(rank: int, result):
    with st.container(border=True):
        if result.hill_suspected_but_needs_crib:
            st.markdown(f"**#{rank} · {result.cipher_display_name}**")
            st.caption(
                f"Classifier says {result.classifier_probability:.0%} likely — but Hill has no "
                f"blind ciphertext-only attack by design. Use the crib tool below."
            )
            return

        # ── Header row ──────────────────────────────────────────────
        st.markdown(f"**#{rank} · {result.cipher_display_name}**  "
                    f"·  *{result.cipher_family}*")

        # ── Two clearly separate signals ─────────────────────────────
        col_clf, col_sol = st.columns(2)

        col_clf.metric(
            label="Classifier probability",
            value=f"{result.classifier_probability:.0%}",
            help=(
                "How confident the ML classifier is that THIS specific cipher produced "
                "the ciphertext — out of 15 possible ciphers. 17% for one cipher out of "
                "15 options (6.7% uniform) is actually a meaningful signal. Ciphers in the "
                "same family often share this probability because they look statistically "
                "identical ciphertext-only (e.g. Substitution and Keyword both relabel "
                "letters with no other distinguishing trace)."
            ),
        )

        col_sol.metric(
            label="Solver method",
            value=result.method.replace("+", " + ").replace("_", " "),
            help=(
                "The actual algorithm used to search for the key once a cipher type was "
                "proposed — brute_force (exhaustive, exact), hill_climbing (heuristic search "
                "over a large key space), or kasiski+ic_scan+frequency (two-stage attack for "
                "polyalphabetic ciphers)."
            ),
        )

        # Solver reliability badge — a separate thing from classifier probability
        st.caption(SOLVER_BADGE.get(result.confidence, result.confidence))

        # ── Decrypted output ─────────────────────────────────────────
        st.text_area(
            "Decrypted text", result.plaintext, height=100, disabled=True,
            key=f"result_{rank}_{result.cipher_slug}",
            label_visibility="collapsed",
        )
        st.code(f"Recovered key: {result.key}", language=None)

        # ── Explainability ───────────────────────────────────────────
        with st.expander("❓ Why this ranking?"):
            for step in explain_result(result):
                st.markdown(f"- {step}")



def _render_frequency_chart(analysis):
    df = pd.DataFrame({
        "Observed (this text)": analysis.observed_letter_freq,
        "Expected (English)": analysis.expected_letter_freq,
    })
    st.bar_chart(df)


def _render_analysis_stats(analysis):
    c1, c2, c3 = st.columns(3)
    c1.metric("Index of Coincidence", f"{analysis.index_of_coincidence:.4f}")
    c2.metric("Chi-squared", f"{analysis.chi_squared:.1f}")
    c3.metric("Word ratio", f"{analysis.word_ratio:.0%}")

    st.caption(
        "**Index of Coincidence** — ~0.067 = monoalphabetic substitution or plain English; "
        "~0.038–0.050 = polyalphabetic; ~0.0385 = uniformly random. "
        "**Chi-squared** — distance from expected English letter frequencies; near 0 means the "
        "letters kept their real identities (points to transposition or plain English), high means "
        "they were relabeled (points to a substitution-family cipher)."
    )

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Bigram LL", f"{analysis.bigram_log_likelihood:.2f}")
    c5.metric("Quadgram LL", f"{analysis.quadgram_log_likelihood:.2f}")
    c6.metric("Digit ratio", f"{analysis.digit_ratio:.0%}")
    c7.metric("Avg token length", f"{analysis.avg_token_length:.1f}")
    st.caption(
        "A high digit ratio (>80%) is a near-certain Polybius signature. An average token length "
        "very close to 2.0 with almost no variation is a near-certain Playfair signature."
    )


st.title("🔐 CryptoAI: Classical Cipher Lab")
st.caption(
    "Automatic detection + cracking of 15 classical ciphers using real frequency analysis, "
    "Kasiski/Friedman key-length recovery, hill-climbing search, and a trained ML classifier — "
    "not a lookup table. Built as a study aid, not a security tool."
)

tab_decrypt, tab_encrypt, tab_analyze = st.tabs(["🕵️ Decrypt", "✍️ Encrypt", "🔬 Analyze"])

# ---------------------------------------------------------------- Decrypt
with tab_decrypt:
    st.subheader("Paste or upload ciphertext, get the top 3 most likely decryptions")
    uploaded_text = _file_upload_text("dec")
    ciphertext = st.text_area(
        "Ciphertext", value=uploaded_text or "", height=140,
        placeholder="Paste an encrypted message here — the app will guess which of the 15 "
                    "ciphers produced it and attempt to crack it automatically.",
    )

    if st.button("🔍 Crack it", type="primary", disabled=not ciphertext.strip()):
        try:
            with st.spinner("Classifying, then actually attempting to crack the top candidates…"):
                results = detect_and_solve(ciphertext, top_k=3)
        except FileNotFoundError:
            st.error(
                "No trained model found yet. From the project root, run:\n\n"
                "`python -m ml.cipher_detector`\n\n(requires `pip install -e \".[dev]\"` once, for training-data generation)."
            )
            results = []

        if results:
            st.success(f"Attempted {len(results)} candidates — best guesses below, ranked by how "
                       f"English the actual decrypted text turned out to be.")
            for i, r in enumerate(results, start=1):
                _render_result_card(i, r)
        elif ciphertext.strip():
            st.warning("Couldn't produce a confident guess for this input.")

    with st.expander("🧩 Know the plaintext starts with something specific? Try the Hill cipher directly"):
        st.caption(
            "Hill cipher has no reliable ciphertext-only attack (see the project README for why) — "
            "real Hill cryptanalysis needs a known-plaintext \"crib\". If you suspect Hill and can "
            "guess how the message starts, try it here."
        )
        hill_ciphertext = st.text_area("Ciphertext", key="hill_ct", height=80)
        crib = st.text_input("Known/guessed plaintext at the very start of the message", key="hill_crib")
        block_size = st.selectbox("Block size (try 2 first)", [2, 3, 4], key="hill_n")
        if st.button("Attempt crib attack", disabled=not (hill_ciphertext.strip() and crib.strip())):
            from solvers.hill_solver import crack_known_plaintext
            try:
                result = crack_known_plaintext(hill_ciphertext, crib, block_size=block_size)
                st.success(f"Recovered key: `{result.key}`")
                st.text_area("Decrypted text", result.plaintext, height=100, disabled=True)
            except ValueError as e:
                st.error(str(e))

# ---------------------------------------------------------------- Encrypt
with tab_encrypt:
    st.subheader("Pick a cipher, encrypt a message")
    uploaded_plaintext = _file_upload_text("enc")
    plaintext = st.text_area("Plaintext", value=uploaded_plaintext or "", height=120,
                              placeholder="Type a message to encrypt…", key="enc_plaintext")

    families = list_families()
    family_choice = st.selectbox("Cipher family", list(FAMILY_LABELS.keys()), format_func=lambda f: FAMILY_LABELS[f])
    cipher_choice = st.selectbox(
        "Cipher — start typing to search", [s.slug for s in families[family_choice]],
        format_func=lambda slug: get_cipher(slug).display_name,
    )
    spec = get_cipher(cipher_choice)
    st.caption(f"📜 {spec.description} *({spec.era})*")

    key = _key_input_widget(spec, key_prefix="enc")

    if st.button("🔒 Encrypt", type="primary", disabled=not plaintext.strip()):
        try:
            ciphertext_out = spec.encrypt(plaintext, key)
            st.text_area("Ciphertext", ciphertext_out, height=120, disabled=False)
            st.code(ciphertext_out, language=None)
        except Exception as e:
            st.error(f"Couldn't encrypt with that key: {e}")

# ---------------------------------------------------------------- Analyze
with tab_analyze:
    st.subheader("Statistical profile of any text")
    st.caption(
        "The same signals every solver and the classifier are built on, shown directly — "
        "useful for sanity-checking a result, or just exploring how a cipher's output looks "
        "statistically. Works on ciphertext, plaintext, or anything else."
    )
    uploaded_analyze_text = _file_upload_text("ana")
    analyze_input = st.text_area("Text to analyze", value=uploaded_analyze_text or "", height=140, key="analyze_input")

    if analyze_input.strip():
        analysis = analyze_text(analyze_input)
        _render_analysis_stats(analysis)
        st.markdown("**Letter frequency: observed vs. expected English**")
        _render_frequency_chart(analysis)

st.divider()
st.caption(
    "Educational project — classical ciphers are not secure and this tool is not meant to protect "
    "anything real. See PROJECT_QA.md in the repo for the full reasoning behind that framing."
)
