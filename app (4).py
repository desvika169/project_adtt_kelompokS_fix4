import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import io
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Analisis Sentimen BUMI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0F1117; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E2130 0%, #252A3D 100%);
        border: 1px solid #2E3450;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 12px;
        color: #8B92A5;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #5A8FBF;
        margin-top: 4px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #5A8FBF22 0%, transparent 100%);
        border-left: 3px solid #5A8FBF;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
        font-size: 16px;
        font-weight: 600;
        color: #FFFFFF;
    }
    
    /* Sentiment badges */
    .badge-pos { background: #1B3A2D; color: #4ADE80; border: 1px solid #4ADE8055; 
                 padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-neg { background: #3A1B1B; color: #F87171; border: 1px solid #F8717155;
                 padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .badge-neu { background: #2A2A1B; color: #FBBF24; border: 1px solid #FBBF2455;
                 padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    
    /* Info box */
    .info-box {
        background: #1A2744;
        border: 1px solid #2E4A7A;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 14px;
        color: #B0C4DE;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1A1D2E;
        border-right: 1px solid #2E3450;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E2130;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #8B92A5;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #5A8FBF !important;
        color: white !important;
    }
    
    /* Divider */
    hr { border-color: #2E3450; margin: 24px 0; }

    /* Scrollable table */
    .scroll-table {
        max-height: 320px;
        overflow-y: auto;
        border-radius: 8px;
        border: 1px solid #2E3450;
    }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
COLORS = {
    "blue":   "#5A8FBF",
    "orange": "#E07B54",
    "green":  "#6DBF6D",
    "red":    "#E05454",
    "purple": "#9B7FD4",
    "yellow": "#F0C040",
}
PALETTE = [COLORS["blue"], COLORS["orange"], COLORS["green"],
           COLORS["red"], COLORS["purple"], COLORS["yellow"]]

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  "#1E2130",
        "axes.facecolor":    "#1E2130",
        "axes.edgecolor":    "#2E3450",
        "axes.labelcolor":   "#8B92A5",
        "xtick.color":       "#8B92A5",
        "ytick.color":       "#8B92A5",
        "text.color":        "#FFFFFF",
        "grid.color":        "#2E3450",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "axes.titlesize":    13,
        "axes.titlecolor":   "#FFFFFF",
        "axes.titlepad":     12,
    })

set_plot_style()

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#1E2130")
    buf.seek(0)
    return buf

# ─── PREPROCESSING ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # ── clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text

    df["clean_text"] = df["komentar"].apply(clean_text)

    # ── tokenisasi
    df["tokens"] = df["clean_text"].apply(lambda x: re.findall(r"\b\w+\b", x))

    # ── normalisasi slang
    slang_dict = {
        "ga":"tidak","gak":"tidak","nggak":"tidak","ngga":"tidak",
        "enggak":"tidak","tdk":"tidak","g":"tidak","nggk":"tidak",
        "kerass":"keras","holdd":"hold","avg":"average","giniin":"seperti"
    }
    df["normalized"] = df["tokens"].apply(
        lambda toks: [slang_dict.get(w, w) for w in toks]
    )

    # ── stopword removal
    stopwords_custom = set([
        "dan","di","ke","dari","yang","untuk","ini","itu","aja","sih","nya","bukan",
        "udah","sudah","saya","aku","yg","karena","dengan","dalam","ada","jadi","buat",
        "pada","juga","kalo","ny","lah","si","bgt","kalau","sama","masih","lagi","moga",
        "semoga","bisa","bikin","krn","yah","bikinan","cuma","cuman","saja","loh","kok",
        "tapi","pake","tp","atau","far","blm","kira","soal","g","ku","pdhl","masa","ya",
        "jg","seperti","ternyata","bln","terus","sdh","gitu","begitu","yaa","tahun","ng",
        "padan","klo","jd","biar","apa","gw","halo","huft","oleh","nih","gak","ga","gaa",
        "nggak","ngga","enggak","deh","dong","sih","weh","wah","ih","ah","oh","eh","hmm",
        "hm","haha","hehe","hihi","wkwk","wkwkwk","banget","bgt","emang","memang","mah",
        "nah","kan","kn","tuh","lg","udh","dah","udah","skrg","sekarang","br","baru","mau",
        "mo","mw","yuk","yukk","ok","oke","okay","okey","iya","iyaa","iyaaa","tak","jangan",
        "belum","dgn","dg","sm","utk","dr","pd","karna","krna","ttg","kak",
    ])
    negation_words = {'tidak','tak','bukan','jangan','belum','ga','gak','nggak','ngga','enggak','g','nggk'}
    all_stopwords = stopwords_custom - negation_words

    df["no_stopword"] = df["normalized"].apply(
        lambda toks: [t for t in toks if t not in all_stopwords and t.strip() != ""]
    )

    # ── negation handling
    def handle_negation(tokens):
        result, skip = [], 0
        for i in range(len(tokens)):
            if skip: skip -= 1; continue
            if tokens[i] in negation_words and i+1 < len(tokens):
                result.append(tokens[i] + "_" + tokens[i+1]); skip = 1
            else:
                result.append(tokens[i])
        return result
    df["negation"] = df["no_stopword"].apply(handle_negation)

    # ── simple stemming (suffix stripping — Sastrawi not available without install)
    SUFFIXES = ["kan","an","i","lah","pun","ku","mu","nya","kah","tah"]
    PREFIXES = ["me","di","ke","se","ter","ber","pe","meng","meny","mem","men","mempel","pem","pen","peng","peny","per","trans"]

    def simple_stem(word):
        for suf in SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) > 2:
                return word[:-len(suf)]
        for pre in PREFIXES:
            if word.startswith(pre) and len(word) - len(pre) > 2:
                return word[len(pre):]
        return word

    df["stems"]     = df["negation"].apply(lambda toks: [simple_stem(w) for w in toks])
    df["stem_text"] = df["stems"].apply(lambda x: " ".join(x))

    # ── stats for before/after
    df["length"]      = df["stem_text"].astype(str).apply(len)
    df["token_count"] = df["stems"].apply(len)
    df["raw_length"]  = df["komentar"].astype(str).apply(len)

    # ── labeling via leksikon (simplified: keyword-based)
    pos_words = set(["naik","bagus","untung","profit","beli","bullish","kuat","mantap",
                     "positif","oke","optimis","rally","hijau","cuan","up","gain","bagus",
                     "rebound","good","great","strong","berkembang","maju","sukses"])
    neg_words = set(["turun","rugi","jual","bearish","jelek","negatif","anjlok","merah",
                     "loss","down","lemah","buruk","parah","crash","drop","koreksi","bangkrut",
                     "susah","jeblok","hancur","parah","jelek"])

    def score_text(tokens):
        s = 0
        for w in tokens:
            if w in pos_words: s += 1
            elif w in neg_words: s -= 1
        return s

    df["sentiment_score"] = df["stems"].apply(score_text)
    df["label"] = df["sentiment_score"].apply(
        lambda s: 1 if s > 0 else (0 if s < 0 else 2)
    )

    # ── date
    if "tanggal" in df.columns:
        df["tanggal_dt"] = pd.to_datetime(df["tanggal"], errors="coerce", utc=True)

    return df

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:36px;'>📈</div>
        <div style='font-size:18px; font-weight:700; color:#FFFFFF; margin-top:8px;'>BUMI Sentiment</div>
        <div style='font-size:11px; color:#8B92A5; margin-top:4px;'>Analisis Komentar Stockbit</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    uploaded = st.file_uploader(
        "📂 Upload CSV komentar BUMI",
        type=["csv"],
        help="Upload file CSV hasil scraping Stockbit"
    )
    
    st.divider()
    st.markdown("""
    <div style='font-size:12px; color:#8B92A5;'>
    <b style='color:#FFFFFF'>📋 Alur Analisis</b><br><br>
    1️⃣ Web Scraping Stockbit<br>
    2️⃣ Text Preprocessing<br>
    3️⃣ Exploratory Data Analysis<br>
    4️⃣ Sentiment Labeling<br>
    5️⃣ Klasifikasi Naive Bayes<br>
    6️⃣ Insight & Rekomendasi
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("""
    <div style='font-size:11px; color:#8B92A5; text-align:center;'>
    Proyek ADTT · Analisis Teks<br>Sentimen Saham BUMI
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN CONTENT ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px 0;'>
    <h1 style='font-size:28px; font-weight:700; margin:0; color:#FFFFFF;'>
        📊 Dashboard Analisis Sentimen Saham BUMI
    </h1>
    <p style='font-size:14px; color:#8B92A5; margin-top:6px;'>
        Analisis komentar investor di Stockbit — Preprocessing · EDA · Naive Bayes Classification
    </p>
</div>
""", unsafe_allow_html=True)

# ── no file yet
if uploaded is None:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1E2130,#252A3D); border:1px dashed #2E3450; 
                border-radius:16px; padding:60px; text-align:center; margin-top:20px;'>
        <div style='font-size:48px;'>📂</div>
        <h3 style='color:#FFFFFF; margin:16px 0 8px 0;'>Upload CSV untuk Mulai</h3>
        <p style='color:#8B92A5; font-size:14px; max-width:400px; margin:0 auto;'>
            Upload file CSV hasil scraping komentar BUMI dari Stockbit di sidebar kiri 
            untuk melihat seluruh analisis.
        </p>
        <div style='margin-top:24px; font-size:12px; color:#5A8FBF;'>
            Format kolom: stream_id · username · fullname · komentar · tanggal · likes · replies · topics
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── load data
with st.spinner("⏳ Memproses data..."):
    df = load_and_preprocess(uploaded)

# ══════════════════════════════════════════════════════════════════════════════
# TAB NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🔧 Preprocessing",
    "🔍 EDA",
    "🤖 Klasifikasi",
    "💡 Insight",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI Cards
    total   = len(df)
    pos_pct = (df["label"]==1).mean()*100
    neg_pct = (df["label"]==0).mean()*100
    neu_pct = (df["label"]==2).mean()*100
    avg_lik = df["likes"].mean() if "likes" in df.columns else 0
    users   = df["username"].nunique() if "username" in df.columns else "-"

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "Total Komentar", f"{total:,}", "data valid"),
        (c2, "Sentimen Positif", f"{pos_pct:.1f}%", f"{(df['label']==1).sum():,} komentar"),
        (c3, "Sentimen Negatif", f"{neg_pct:.1f}%", f"{(df['label']==0).sum():,} komentar"),
        (c4, "Avg Likes", f"{avg_lik:.1f}", "per komentar"),
        (c5, "Total User", f"{users}", "unik"),
    ]
    for col, label, val, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sentiment donut + tren
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('<div class="section-header">🥧 Distribusi Sentimen</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        counts = [
            (df["label"]==1).sum(),
            (df["label"]==0).sum(),
            (df["label"]==2).sum(),
        ]
        labels_pie = ["Positif", "Negatif", "Netral"]
        colors_pie = [COLORS["green"], COLORS["orange"], COLORS["yellow"]]
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels_pie, autopct="%1.1f%%",
            colors=colors_pie, startangle=140,
            wedgeprops=dict(width=0.55, edgecolor="#1E2130", linewidth=2),
            pctdistance=0.75,
        )
        for at in autotexts: at.set(fontsize=10, color="white", fontweight="bold")
        for t in texts:      t.set(fontsize=10, color="#CCCCCC")
        ax.set_title("Komposisi Sentimen BUMI", fontsize=13, color="white")
        fig.patch.set_facecolor("#1E2130")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown('<div class="section-header">📅 Tren Komentar per Hari</div>', unsafe_allow_html=True)
        if "tanggal_dt" in df.columns and df["tanggal_dt"].notna().any():
            tren = df.set_index("tanggal_dt").resample("D")["stream_id"].count()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(tren.index, tren.values, color=COLORS["blue"], linewidth=2, marker="o", markersize=3)
            ax.fill_between(tren.index, tren.values, alpha=0.15, color=COLORS["blue"])
            peak = tren.idxmax()
            ax.annotate(f"Puncak: {tren.max()}",
                        xy=(peak, tren.max()),
                        xytext=(peak, tren.max()*1.12),
                        ha="center", fontsize=9, color=COLORS["orange"],
                        arrowprops=dict(arrowstyle="->", color=COLORS["orange"]))
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Jumlah Komentar")
            ax.set_title("Aktivitas Komentar BUMI per Hari")
            ax.grid(True, alpha=0.3)
            fig.patch.set_facecolor("#1E2130")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Kolom tanggal tidak tersedia atau tidak dapat di-parse.")

    # Sample data
    st.markdown('<div class="section-header">📋 Sample Data</div>', unsafe_allow_html=True)
    sample = df[["username","komentar","likes","replies","label"]].head(10).copy()
    sample["label"] = sample["label"].map({1:"✅ Positif", 0:"❌ Negatif", 2:"⚪ Netral"})
    st.dataframe(sample, use_container_width=True, height=280)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">⚙️ Pipeline Preprocessing</div>', unsafe_allow_html=True)

    # Pipeline steps
    steps = [
        ("1️⃣", "Case Folding", "Semua teks diubah ke huruf kecil"),
        ("2️⃣", "Remove Angka & Simbol", "Digit dan tanda baca dihapus"),
        ("3️⃣", "Normalisasi Slang", "Kata gaul → kata baku (gak→tidak, dll)"),
        ("4️⃣", "Tokenisasi", "Teks dipecah menjadi token kata"),
        ("5️⃣", "Stopword Removal", "Kata tidak bermakna dihapus"),
        ("6️⃣", "Negation Handling", "Kata negasi digabung (tidak_bagus)"),
        ("7️⃣", "Stemming", "Kata dikembalikan ke bentuk dasar"),
    ]
    cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:#1E2130;border:1px solid #2E3450;border-radius:10px;
                        padding:14px 10px;text-align:center;height:120px;'>
                <div style='font-size:20px;'>{num}</div>
                <div style='font-size:12px;font-weight:600;color:#FFFFFF;margin:6px 0 4px;'>{title}</div>
                <div style='font-size:10px;color:#8B92A5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Before vs After comparison
    st.markdown('<div class="section-header">🔁 Perbandingan Sebelum & Sesudah Preprocessing</div>', unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("**🔴 Sebelum Preprocessing**")
        st.dataframe(
            df[["komentar"]].rename(columns={"komentar":"Teks Asli"}).head(15),
            use_container_width=True, height=380
        )
    with col_y:
        st.markdown("**🟢 Sesudah Preprocessing**")
        st.dataframe(
            df[["stem_text"]].rename(columns={"stem_text":"Teks Bersih"}).head(15),
            use_container_width=True, height=380
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats comparison charts
    st.markdown('<div class="section-header">📉 Statistik Panjang Teks: Sebelum vs Sesudah</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        sns.histplot(df["raw_length"], bins=30, ax=axes[0], color=COLORS["orange"], alpha=0.8)
        axes[0].set_title("Panjang Karakter (Raw)")
        axes[0].set_xlabel("Karakter")
        axes[0].set_ylabel("Jumlah")
        sns.histplot(df["length"], bins=30, ax=axes[1], color=COLORS["blue"], alpha=0.8)
        axes[1].set_title("Panjang Karakter (Bersih)")
        axes[1].set_xlabel("Karakter")
        axes[1].set_ylabel("Jumlah")
        plt.tight_layout()
        fig.patch.set_facecolor("#1E2130")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        # Reduction stats
        avg_before = df["raw_length"].mean()
        avg_after  = df["length"].mean()
        avg_tok    = df["token_count"].mean()
        reduction  = (1 - avg_after/avg_before) * 100

        st.markdown(f"""
        <div style='background:#1E2130;border:1px solid #2E3450;border-radius:12px;padding:24px;margin-top:8px;'>
            <div style='font-size:14px;font-weight:600;color:#FFFFFF;margin-bottom:16px;'>📊 Ringkasan Preprocessing</div>
            <table style='width:100%;font-size:13px;color:#B0C4DE;border-collapse:collapse;'>
            <tr style='border-bottom:1px solid #2E3450;'>
                <td style='padding:8px 0;color:#8B92A5;'>Rata-rata panjang SEBELUM</td>
                <td style='text-align:right;color:#E07B54;font-weight:600;'>{avg_before:.0f} char</td>
            </tr>
            <tr style='border-bottom:1px solid #2E3450;'>
                <td style='padding:8px 0;color:#8B92A5;'>Rata-rata panjang SESUDAH</td>
                <td style='text-align:right;color:#5A8FBF;font-weight:600;'>{avg_after:.0f} char</td>
            </tr>
            <tr style='border-bottom:1px solid #2E3450;'>
                <td style='padding:8px 0;color:#8B92A5;'>Rata-rata jumlah token</td>
                <td style='text-align:right;color:#6DBF6D;font-weight:600;'>{avg_tok:.1f} token</td>
            </tr>
            <tr>
                <td style='padding:8px 0;color:#8B92A5;'>Reduksi teks</td>
                <td style='text-align:right;color:#F0C040;font-weight:600;'>{reduction:.1f}%</td>
            </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Token distribution
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.histplot(df["token_count"], bins=25, ax=ax2, color=COLORS["green"], alpha=0.8)
        ax2.set_title("Distribusi Jumlah Token per Komentar")
        ax2.set_xlabel("Jumlah Token")
        ax2.set_ylabel("Frekuensi")
        ax2.grid(True, alpha=0.3)
        fig2.patch.set_facecolor("#1E2130")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    # Word frequency
    st.markdown('<div class="section-header">🔤 Frekuensi Kata</div>', unsafe_allow_html=True)

    text  = " ".join(df["stem_text"].dropna().astype(str))
    words = text.lower().split()
    word_counts = Counter(words)

    col1, col2 = st.columns(2)

    with col1:
        n_words = st.slider("Tampilkan top N kata:", 5, 30, 10, key="freq_slider")
        common = dict(word_counts.most_common(n_words))
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(list(common.keys())[::-1], list(common.values())[::-1], color=COLORS["blue"])
        ax.bar_label(bars, padding=4, color="#CCCCCC", fontsize=9)
        ax.set_xlabel("Frekuensi")
        ax.set_title(f"Top {n_words} Kata Paling Sering Muncul")
        ax.grid(True, axis="x", alpha=0.3)
        fig.patch.set_facecolor("#1E2130")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("**☁️ Word Cloud**")
        if text.strip():
            wc = WordCloud(
                width=700, height=380,
                background_color="#1E2130",
                colormap="Blues",
                max_words=80,
            ).generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(7, 4))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            ax_wc.set_title("WordCloud Komentar BUMI")
            fig_wc.patch.set_facecolor("#1E2130")
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)

    # Likes & Replies
    st.markdown('<div class="section-header">❤️ Distribusi Likes & Replies</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    for col, colname, color, label in [
        (col3, "likes",   COLORS["blue"],   "Likes"),
        (col4, "replies", COLORS["orange"], "Replies"),
    ]:
        with col:
            if colname in df.columns:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.histplot(df[colname], bins=30, ax=ax, color=color, alpha=0.85)
                ax.set_xlabel(label)
                ax.set_ylabel("Jumlah Komentar")
                ax.set_title(f"Distribusi {label}")
                ax.grid(True, alpha=0.3)
                fig.patch.set_facecolor("#1E2130")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # Top users
    st.markdown('<div class="section-header">👤 Top 10 User Paling Aktif</div>', unsafe_allow_html=True)

    if "username" in df.columns:
        top_users = df["username"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(top_users.index[::-1], top_users.values[::-1], color=COLORS["purple"])
        ax.bar_label(bars, padding=4, color="#CCCCCC", fontsize=9)
        ax.set_xlabel("Jumlah Komentar")
        ax.set_title("Top 10 User Paling Aktif di Stream BUMI")
        ax.grid(True, axis="x", alpha=0.3)
        fig.patch.set_facecolor("#1E2130")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Bigram
    st.markdown('<div class="section-header">🔗 Analisis Bigram</div>', unsafe_allow_html=True)

    from sklearn.feature_extraction.text import CountVectorizer as CV

    def get_top_bigram(corpus, n=10):
        vec = CV(ngram_range=(2,2), max_features=500)
        vec.fit(corpus)
        bow = vec.transform(corpus)
        sw  = bow.sum(axis=0)
        freq = [(w, sw[0, i]) for w, i in vec.vocabulary_.items()]
        return sorted(freq, key=lambda x: x[1], reverse=True)[:n]

    label_map   = {0:"Negatif", 1:"Positif", 2:"Netral"}
    color_map   = {0:COLORS["orange"], 1:COLORS["blue"], 2:COLORS["yellow"]}
    bigram_tabs = st.tabs(["Semua", "Positif", "Negatif", "Netral"])

    for idx, (tab_bg, label_id) in enumerate(zip(bigram_tabs, [-1, 1, 0, 2])):
        with tab_bg:
            corpus = (df["stem_text"].dropna() if label_id == -1
                      else df[df["label"]==label_id]["stem_text"].dropna())
            if len(corpus) < 2:
                st.info("Data tidak cukup untuk bigram.")
                continue
            bg = get_top_bigram(corpus)
            if not bg: continue
            bg_df = pd.DataFrame(bg, columns=["Bigram","Frekuensi"])
            color = COLORS["green"] if label_id == -1 else color_map.get(label_id, COLORS["blue"])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(bg_df["Bigram"][::-1], bg_df["Frekuensi"][::-1], color=color)
            title = "Semua" if label_id == -1 else label_map.get(label_id,"")
            ax.set_title(f"Top 10 Bigram — {title}")
            ax.set_xlabel("Frekuensi")
            ax.grid(True, axis="x", alpha=0.3)
            fig.patch.set_facecolor("#1E2130")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — KLASIFIKASI NAIVE BAYES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

    st.markdown('<div class="section-header">🤖 Klasifikasi Naive Bayes</div>', unsafe_allow_html=True)

    df_model = df[df["label"] != 2].copy()
    df_model["label"] = df_model["label"].astype(int)

    if len(df_model) < 10:
        st.warning("Data terlalu sedikit untuk training model.")
    else:
        X = df_model["stem_text"]
        y = df_model["label"]

        test_size = st.slider("Proporsi Data Test:", 0.1, 0.4, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        max_feat = st.slider("Max Fitur TF-IDF:", 200, 2000, 1000, 100)
        tfidf = TfidfVectorizer(max_features=max_feat)
        X_tr  = tfidf.fit_transform(X_train)
        X_te  = tfidf.transform(X_test)

        nb = MultinomialNB()
        nb.fit(X_tr, y_train)
        y_pred = nb.predict(X_te)

        acc = accuracy_score(y_test, y_pred) * 100
        report = classification_report(y_test, y_pred,
                                        target_names=["Negatif","Positif"],
                                        output_dict=True)

        # Metric cards
        r_pos = report.get("Positif", {})
        r_neg = report.get("Negatif", {})

        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        metrics_nb = [
            (mc1, "Accuracy", f"{acc:.2f}%", "keseluruhan"),
            (mc2, "Precision (Pos)", f"{r_pos.get('precision',0)*100:.1f}%", "Naive Bayes"),
            (mc3, "Recall (Pos)", f"{r_pos.get('recall',0)*100:.1f}%", "Naive Bayes"),
            (mc4, "F1-Score (Pos)", f"{r_pos.get('f1-score',0)*100:.1f}%", "Naive Bayes"),
        ]
        for col, label, val, sub in metrics_nb:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**🟦 Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Negatif","Positif"],
                        yticklabels=["Negatif","Positif"],
                        linewidths=1, linecolor="#1E2130",
                        cbar_kws={"shrink":0.8})
            ax.set_title("Confusion Matrix — Naive Bayes")
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            fig.patch.set_facecolor("#1E2130")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_b:
            st.markdown("**📊 Distribusi Hasil Prediksi**")
            pred_s = pd.Series(y_pred).value_counts()
            pred_s.index = ["Positif" if i==1 else "Negatif" for i in pred_s.index]
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            bars = ax2.bar(pred_s.index, pred_s.values,
                           color=[COLORS["blue"],COLORS["orange"]])
            ax2.bar_label(bars, padding=5, color="#CCCCCC", fontweight="bold")
            ax2.set_xlabel("Sentimen")
            ax2.set_ylabel("Jumlah Komentar")
            ax2.set_title("Distribusi Hasil Prediksi")
            ax2.grid(True, axis="y", alpha=0.3)
            fig2.patch.set_facecolor("#1E2130")
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        # Full classification report
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📋 Classification Report Lengkap**")
        report_rows = []
        for cls in ["Negatif","Positif","macro avg","weighted avg"]:
            if cls in report:
                r = report[cls]
                report_rows.append({
                    "Kelas": cls,
                    "Precision": f"{r['precision']:.4f}",
                    "Recall":    f"{r['recall']:.4f}",
                    "F1-Score":  f"{r['f1-score']:.4f}",
                    "Support":   int(r.get("support",0)),
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

        # Data split info
        st.markdown('<div class="section-header">📐 Info Split Data</div>', unsafe_allow_html=True)
        ic1, ic2, ic3 = st.columns(3)
        for col, label, val in [
            (ic1, "Total Data (non-Netral)", len(df_model)),
            (ic2, "Data Train", len(X_train)),
            (ic3, "Data Test",  len(X_test)),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val:,}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INSIGHT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">💡 Insight Utama</div>', unsafe_allow_html=True)

    pos_count = (df["label"]==1).sum()
    neg_count = (df["label"]==0).sum()
    neu_count = (df["label"]==2).sum()
    dominant  = "POSITIF" if pos_count > neg_count else "NEGATIF"
    dom_color = "#4ADE80" if dominant=="POSITIF" else "#F87171"

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1E2130,#252A3D);border:1px solid #2E3450;
                border-radius:14px;padding:24px;margin-bottom:20px;'>
        <h3 style='color:#FFFFFF;margin:0 0 12px 0;'>🎯 Kesimpulan Sentimen BUMI</h3>
        <p style='color:#B0C4DE;font-size:14px;line-height:1.7;'>
        Dari <b style='color:white;'>{len(df):,}</b> komentar investor di Stockbit,
        sentimen yang mendominasi adalah 
        <b style='color:{dom_color};font-size:16px;'>{dominant}</b>
        dengan <b style='color:{dom_color};'>{max(pos_count,neg_count):,}</b> komentar 
        ({max(pos_pct,neg_pct):.1f}% dari total).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Insight cards
    insight_cols = st.columns(3)
    insights = [
        ("📈", "Sentimen Positif", f"{pos_count:,} komentar ({pos_pct:.1f}%)",
         "Investor cenderung optimis terhadap prospek BUMI, didorong ekspektasi kenaikan harga batu bara dan ekspansi bisnis.",
         COLORS["green"]),
        ("📉", "Sentimen Negatif", f"{neg_count:,} komentar ({neg_pct:.1f}%)",
         "Kekhawatiran investor terkait kondisi pasar global, fluktuasi harga komoditas, dan isu fundamental perusahaan.",
         COLORS["orange"]),
        ("⚪", "Sentimen Netral", f"{neu_count:,} komentar ({neu_pct:.1f}%)",
         "Komentar informatif atau diskusi teknikal tanpa bias sentimen positif/negatif yang jelas.",
         COLORS["yellow"]),
    ]
    for col, (icon, title, stat, desc, color) in zip(insight_cols, insights):
        with col:
            st.markdown(f"""
            <div style='background:#1E2130;border:1px solid {color}44;border-top:3px solid {color};
                        border-radius:12px;padding:20px;height:200px;'>
                <div style='font-size:24px;'>{icon}</div>
                <div style='font-size:14px;font-weight:700;color:#FFFFFF;margin:8px 0 4px;'>{title}</div>
                <div style='font-size:13px;font-weight:600;color:{color};margin-bottom:10px;'>{stat}</div>
                <div style='font-size:12px;color:#8B92A5;line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top positive & negative keywords
    st.markdown('<div class="section-header">🔑 Kata Kunci per Sentimen</div>', unsafe_allow_html=True)

    k1, k2 = st.columns(2)
    for col, label_id, color, title in [
        (k1, 1, COLORS["green"],  "✅ Kata Kunci Positif"),
        (k2, 0, COLORS["orange"], "❌ Kata Kunci Negatif"),
    ]:
        with col:
            subset = df[df["label"]==label_id]["stem_text"].dropna()
            if len(subset) == 0:
                col.info("Tidak ada data.")
                continue
            sub_words = " ".join(subset).split()
            sub_counts = Counter(sub_words).most_common(12)
            if not sub_counts: continue
            sw_df = pd.DataFrame(sub_counts, columns=["Kata","Frekuensi"])
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.barh(sw_df["Kata"][::-1], sw_df["Frekuensi"][::-1], color=color, alpha=0.85)
            ax.set_title(title, color="#FFFFFF")
            ax.set_xlabel("Frekuensi")
            ax.grid(True, axis="x", alpha=0.3)
            fig.patch.set_facecolor("#1E2130")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Engagement analysis
    st.markdown('<div class="section-header">❤️ Analisis Engagement per Sentimen</div>', unsafe_allow_html=True)

    if "likes" in df.columns and "replies" in df.columns:
        eng = df[df["label"]!=2].groupby("label")[["likes","replies"]].mean().round(2)
        eng.index = [("Negatif" if i==0 else "Positif") for i in eng.index]
        eng_cols = st.columns(2)
        for col, metric, color in [
            (eng_cols[0], "likes",   COLORS["blue"]),
            (eng_cols[1], "replies", COLORS["orange"]),
        ]:
            with col:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                bars = ax.bar(eng.index, eng[metric], color=[COLORS["orange"],COLORS["blue"]])
                ax.bar_label(bars, padding=5, color="#CCCCCC", fontweight="bold")
                ax.set_ylabel(f"Rata-rata {metric.capitalize()}")
                ax.set_title(f"Avg {metric.capitalize()} per Sentimen")
                ax.grid(True, axis="y", alpha=0.3)
                fig.patch.set_facecolor("#1E2130")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # Rekomendasi
    st.markdown('<div class="section-header">📌 Rekomendasi & Kesimpulan</div>', unsafe_allow_html=True)

    rekoms = [
        ("🔍", "Monitor Sentimen Real-time",
         "Lakukan scraping berkala untuk memantau perubahan sentimen investor BUMI yang bisa menjadi indikator pergerakan harga saham."),
        ("📊", "Analisis Event-Driven",
         "Korelasikan lonjakan komentar (peak activity) dengan event korporasi atau berita makroekonomi untuk identifikasi katalis sentimen."),
        ("🤖", "Tingkatkan Akurasi Model",
         "Gunakan model lanjutan (SVM, BERT berbasis bahasa Indonesia) dan perluas kamus leksikon untuk meningkatkan kualitas klasifikasi sentimen."),
        ("📈", "Deteksi Manipulasi Pasar",
         "Analisis user dengan aktivitas abnormal tinggi untuk mendeteksi potensi pump-and-dump atau manipulasi sentimen di forum saham."),
    ]
    for icon, title, desc in rekoms:
        st.markdown(f"""
        <div style='background:#1E2130;border:1px solid #2E3450;border-radius:10px;
                    padding:16px 20px;margin-bottom:10px;display:flex;align-items:flex-start;gap:14px;'>
            <div style='font-size:22px;flex-shrink:0;'>{icon}</div>
            <div>
                <div style='font-size:14px;font-weight:600;color:#FFFFFF;margin-bottom:4px;'>{title}</div>
                <div style='font-size:13px;color:#8B92A5;line-height:1.5;'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)
