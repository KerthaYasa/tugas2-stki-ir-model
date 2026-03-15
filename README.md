# Sistem Temu Kembali Informasi
Tugas STKI — Extended Boolean Model

## Cara Menjalankan
1. Install library yang dibutuhkan:
   pip install -r requirements.txt

2. Jalankan aplikasi:
   streamlit run app.py

3. Buka browser di: http://localhost:8501

## Struktur Folder
```text
tugas-stki/
├── corpus/          ← 7 dokumen teks
├── preprocessing.py ← tokenisasi, stemming
├── indexing.py      ← incidence matrix & inverted index
├── ir_model.py      ← extended boolean model
├── app.py           ← UI Streamlit
├── requirements.txt ← daftar library
└── README.md        ← dokumentasi
```

## Fitur
- Pencarian dengan operator AND, OR, NOT, dan kurung ()
- Incidence Matrix
- Inverted Index
