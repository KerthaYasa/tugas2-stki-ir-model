import streamlit as st
import pandas as pd
import os
from preprocessing import load_corpus
from indexing import build_vocabulary, build_incidence_matrix, build_inverted_index_full
from ir_model import compute_tf_normalized, search

st.set_page_config(page_title="IR System — STKI", page_icon="🔎", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #f0f5f2; }

/* Panel kiri — hijau muda */
[data-testid="column"]:first-child > div:first-child {
    background: #e4f0ea;
    border: 1px solid #c5ddd0;
    border-radius: 14px;
    padding: 20px !important;
}

/* Panel kanan — putih */
[data-testid="column"]:last-child > div:first-child {
    background: #ffffff;
    border: 1px solid #dde8e2;
    border-radius: 14px;
    padding: 20px !important;
}

/* Input */
.stTextInput input {
    background: #fff !important;
    border: 1.5px solid #c5ddd0 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.83rem !important;
    color: #1c2b24 !important;
}
.stTextInput input:focus {
    border-color: #2d8c5e !important;
    box-shadow: 0 0 0 3px rgba(45,140,94,0.1) !important;
}

/* Semua tombol */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    border-radius: 7px !important;
    border: 1.5px solid #c5ddd0 !important;
    background: #fff !important;
    color: #2d8c5e !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #2d8c5e !important;
    color: #fff !important;
    border-color: #2d8c5e !important;
}

/* Tombol cari — selalu hijau */
div[data-testid="column"]:first-child .stButton:last-of-type > button {
    background: #2d8c5e !important;
    color: #fff !important;
    border-color: #2d8c5e !important;
    font-weight: 600 !important;
    width: 100% !important;
}

/* Expander hasil */
[data-testid="stExpander"] {
    background: #f7fbf8 !important;
    border: 1px solid #dde8e2 !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #1c2b24 !important;
}
[data-testid="stExpander"] summary:hover { background: #e4f0ea !important; }

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #2d8c5e, #52c48a) !important;
    border-radius: 99px !important;
}
.stProgress > div {
    background: #d4ebe0 !important;
    border-radius: 99px !important;
    height: 6px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #fff !important;
    border: 1px solid #dde8e2 !important;
    border-radius: 9px !important;
    padding: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #7a9488 !important;
    border-radius: 6px !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #2d8c5e !important;
    color: #fff !important;
}

/* Chip term */
.chip-hit {
    display: inline-block;
    background: #e4f0ea; border: 1px solid #b6ddc8;
    color: #2d8c5e; font-family: 'DM Mono', monospace;
    font-size: 0.68rem; padding: 2px 10px;
    border-radius: 99px; margin: 2px;
}
.chip-miss {
    display: inline-block;
    background: #fdecea; border: 1px solid #f5b7b1;
    color: #c0392b; font-family: 'DM Mono', monospace;
    font-size: 0.68rem; padding: 2px 10px;
    border-radius: 99px; margin: 2px;
}

/* Doc box */
.doc-box {
    background: #f7fbf8;
    border-left: 3px solid #2d8c5e;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.86rem;
    line-height: 1.75;
    color: #2c3e35;
    margin-top: 6px;
}

hr { border-color: #dde8e2 !important; margin: 10px 0 !important; }
.stCaption { font-family: 'DM Mono', monospace !important; font-size: 0.67rem !important; color: #7a9488 !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    corpus     = load_corpus("corpus")
    vocab      = build_vocabulary(corpus)
    inc_matrix = build_incidence_matrix(corpus, vocab)
    inv_idx    = build_inverted_index_full(corpus)
    tf_norm    = compute_tf_normalized(corpus)
    return corpus, vocab, inc_matrix, inv_idx, tf_norm

corpus, vocab, inc_matrix, inv_idx, tf_norm = load_all()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("## 🔎 Sistem Temu Kembali Informasi")
st.caption("Extended Boolean Model · Corpus: Kasus Detektif Ardan · STKI")
st.divider()


# ── LAYOUT DUA KOLOM ─────────────────────────────────────────────────────────
col_kiri, col_kanan = st.columns([1, 2], gap="medium")

# ═══════════════════════════
# KOLOM KIRI — Input & Query
# ═══════════════════════════
with col_kiri:
    st.markdown("#### 🔍 Pencarian")
    st.caption("Gunakan operator AND, OR, NOT dan tanda kurung ( )")

    query_input = st.text_input(
        label="query",
        placeholder="bukti AND kasus AND NOT korban",
        label_visibility="collapsed",
        key="query_main"
    )
    tombol_cari = st.button("Cari Dokumen", use_container_width=True)

    st.divider()

    st.caption("CONTOH QUERY:")
    contoh_list = [
        "bukti AND kasus",
        "bukti AND kasus AND NOT korban",
        "(curi AND bukti) AND NOT bakar",
        "saksi OR korban",
        "ardan AND NOT bunuh",
    ]
    for c in contoh_list:
        if st.button(c, key=f"q_{c}", use_container_width=True):
            query_input = c
            tombol_cari = True

    st.divider()
    st.caption(f"📄 {len(corpus)} dokumen  ·  {len(vocab)} term unik")


# ════════════════════════════
# KOLOM KANAN — Hasil
# ════════════════════════════
with col_kanan:
    st.markdown("#### 📄 Hasil Pencarian")

    if tombol_cari and query_input:
        hasil = search(query_input, corpus, tf_norm)

        if hasil is None:
            st.warning("⚠️ Query tidak valid.")
        else:
            terms, ops, dokumen = hasil
            st.caption(f"query: `{query_input}`  ·  term: {' · '.join(terms)}  ·  {len(dokumen)} dokumen ditemukan")
            st.divider()

            if not dokumen:
                st.info("Tidak ada dokumen yang memenuhi query ini.")
            else:
                skor_max = dokumen[0][1] if dokumen[0][1] > 0 else 1
                badges   = ["🥇", "🥈", "🥉"]

                for rank, (doc, skor) in enumerate(dokumen, start=1):
                    path = os.path.join("corpus", doc)
                    with open(path, 'r', encoding='utf-8') as f:
                        isi = f.read().strip()

                    preview  = isi[:80].replace('\n', ' ') + "..."
                    badge    = badges[rank-1] if rank <= 3 else f"#{rank}"
                    tok      = corpus[doc]

                    with st.expander(
                        f"{badge}  {doc}  —  skor: {skor}   ·   {preview}",
                        expanded=(rank==1)
                    ):
                        st.progress(skor)
                        st.caption(f"skor extended boolean: {skor}  (skala 0.0 — 1.0)")

                        # Chip status term
                        chips = ""
                        for ti, term in enumerate(terms):
                            frek = tok.count(term)
                            op   = ops[ti] if ti < len(ops) else None
                            if op == 'NOT':
                                chips += f'<span class="chip-hit">✓ NOT {term}</span>' if frek == 0 \
                                    else f'<span class="chip-miss">✗ NOT {term} ({frek}x)</span>'
                            else:
                                chips += f'<span class="chip-hit">✓ {term} ({frek}x)</span>' if frek > 0 \
                                    else f'<span class="chip-miss">✗ {term}</span>'
                        st.markdown(chips, unsafe_allow_html=True)
                        st.divider()

                        st.caption("ISI DOKUMEN:")
                        st.markdown(f'<div class="doc-box">{isi}</div>', unsafe_allow_html=True)

    elif tombol_cari and not query_input:
        st.warning("⚠️ Masukkan query terlebih dahulu.")
    else:
        st.markdown("""
        <div style='text-align:center; padding:48px 0; color:#aac4b5;'>
            <div style='font-size:2.5rem'>📭</div>
            <div style='font-family:DM Mono,monospace; font-size:0.72rem;
                        text-transform:uppercase; letter-spacing:0.1em; margin-top:10px;'>
                Belum ada pencarian
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── TABS BAWAH ────────────────────────────────────────────────────────────────
st.divider()
tab1, tab2 = st.tabs(["📊  Incidence Matrix", "📋  Inverted Index"])

# Tab Incidence Matrix
with tab1:
    st.caption(f"{inc_matrix.shape[0]} term × {inc_matrix.shape[1]} dokumen")
    filter_m = st.text_input("Filter term:", placeholder="ketik term...", key="fm")
    filtered = inc_matrix[inc_matrix.index.str.contains(filter_m, case=False)] if filter_m else inc_matrix
    styled   = filtered.style.map(
        lambda v: 'background-color:#e4f0ea;color:#2d8c5e;font-weight:600;text-align:center'
                  if v == 1 else
                  'background-color:#f7fbf8;color:#c0c8c4;text-align:center'
    )
    st.dataframe(styled, use_container_width=True, height=420)

# Tab Inverted Index
with tab2:
    st.caption(f"{len(inv_idx)} term unik")
    filter_i = st.text_input("Filter term:", placeholder="ketik term...", key="fi")
    rows = []
    for term in sorted(inv_idx.keys()):
        entries = inv_idx[term]
        fmt = [f"<{d.replace('.txt','')}, {i['frekuensi']}, {i['posisi']}>" for d, i in entries.items()]
        rows.append({"Term": term, "Inverted List": "  |  ".join(fmt), "Dok": len(entries)})
    df_inv = pd.DataFrame(rows)
    if filter_i:
        df_inv = df_inv[df_inv["Term"].str.contains(filter_i, case=False)]
    st.dataframe(df_inv, use_container_width=True, height=420)