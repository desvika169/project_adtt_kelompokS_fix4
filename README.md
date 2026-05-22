# 📊 Dashboard Analisis Sentimen Saham BUMI

Dashboard interaktif berbasis **Streamlit** untuk menganalisis sentimen komentar investor saham BUMI di platform Stockbit.

## 🚀 Fitur Dashboard

| Tab | Konten |
|-----|--------|
| 🏠 Overview | KPI card, distribusi sentimen (donut chart), tren komentar per hari, sample data |
| 🔧 Preprocessing | Pipeline 7 langkah, perbandingan sebelum/sesudah, statistik reduksi teks |
| 🔍 EDA | Word frequency, word cloud, distribusi likes/replies, top user, analisis bigram per sentimen |
| 🤖 Klasifikasi | Naive Bayes dengan TF-IDF, confusion matrix, classification report, distribusi prediksi |
| 💡 Insight | Kesimpulan sentimen, kata kunci per sentimen, analisis engagement, rekomendasi |

## 🛠️ Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/USERNAME/bumi-sentiment-dashboard.git
cd bumi-sentiment-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Streamlit
```bash
streamlit run app.py
```

### 4. Upload Data
- Buka browser di `http://localhost:8501`
- Upload file CSV hasil scraping Stockbit di sidebar kiri

## 📁 Format File CSV

File CSV harus memiliki kolom-kolom berikut (hasil scraping notebook):

| Kolom | Deskripsi |
|-------|-----------|
| `stream_id` | ID unik komentar |
| `username` | Username Stockbit |
| `fullname` | Nama lengkap user |
| `komentar` | Teks komentar investor |
| `tanggal` | Timestamp komentar |
| `likes` | Jumlah likes |
| `replies` | Jumlah replies |
| `topics` | Topik terkait |

## 🔧 Pipeline Preprocessing

1. **Case Folding** — Konversi ke huruf kecil
2. **Remove Angka & Simbol** — Hapus digit dan tanda baca
3. **Normalisasi Slang** — Mapping kata gaul ke kata baku
4. **Tokenisasi** — Pecah teks menjadi token
5. **Stopword Removal** — Hapus kata tidak bermakna (dengan kamus custom bahasa Indonesia)
6. **Negation Handling** — Gabungkan kata negasi (`tidak_bagus`)
7. **Stemming** — Kembalikan ke bentuk dasar

## 🤖 Model Klasifikasi

- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF (configurable max features)
- **Labeling:** Kamus leksikon sentimen berbasis kata kunci
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
wordcloud>=1.9.0
```

## 📸 Screenshot

Dashboard ini menampilkan tema dark modern dengan visualisasi yang informatif untuk analisis sentimen investor saham BUMI.

## 👩‍💻 Tentang Proyek

Proyek ini merupakan bagian dari tugas **Analisis Data Teks Terstruktur (ADTT)** yang menganalisis sentimen komunitas investor saham BUMI di platform Stockbit menggunakan teknik NLP berbasis bahasa Indonesia.

---
*Built with ❤️ using Streamlit & Scikit-learn*
