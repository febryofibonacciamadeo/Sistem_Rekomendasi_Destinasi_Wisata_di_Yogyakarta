# Sistem Rekomendasi Destinasi Wisata di Yogyakarta

Sistem rekomendasi yang menyarankan destinasi wisata di Yogyakarta kepada pengguna berdasarkan data rating/preferensi wisatawan sebelumnya.

## Deskripsi

Proyek ini membangun sebuah **recommender system** untuk membantu wisatawan menemukan destinasi wisata di Yogyakarta yang sesuai dengan preferensi mereka, menggunakan data rating pengguna terhadap berbagai tempat wisata.

## Struktur Repository

```
Sistem_Rekomendasi_Destinasi_Wisata_di_Yogyakarta/
├── dataset/            # Data destinasi wisata & data pendukung lainnya
├── code.ipynb          # Notebook utama: eksplorasi data, pemodelan, evaluasi
└── rating_final.csv    # Data rating pengguna terhadap destinasi wisata
```

## Alur Kerja (Workflow)

1. **Data Loading:** memuat data destinasi wisata dan rating pengguna (`rating_final.csv`).
2. **Exploratory Data Analysis (EDA):** eksplorasi pola rating dan karakteristik destinasi.
3. **Data Preprocessing:** membersihkan data dan membentuk matriks user-item.
4. **Model Building:** membangun model rekomendasi, misalnya:
   - *Collaborative Filtering* (berbasis kemiripan user/item), dan/atau
   - *Content-Based Filtering* (berbasis kategori/fitur destinasi)
5. **Evaluation:** mengukur kualitas rekomendasi (mis. RMSE, precision@k, atau evaluasi kualitatif).
6. **Rekomendasi:** menghasilkan daftar top-N destinasi wisata untuk pengguna tertentu.

## Library yang Digunakan

- Python
- Jupyter Notebook
- pandas & numpy
- scikit-learn (perhitungan similarity, evaluasi model)

## Cara Menjalankan

1. Clone repository ini:
   ```bash
   git clone https://github.com/febryofibonacciamadeo/Sistem_Rekomendasi_Destinasi_Wisata_di_Yogyakarta.git
   cd Sistem_Rekomendasi_Destinasi_Wisata_di_Yogyakarta
   ```
2. Install dependensi:
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```
3. Jalankan notebook:
   ```bash
   jupyter notebook code.ipynb
   ```

## Hasil

Notebook menghasilkan daftar rekomendasi destinasi wisata Yogyakarta beserta evaluasi performa model. Lihat `code.ipynb` untuk detail metrik dan contoh output rekomendasi.
