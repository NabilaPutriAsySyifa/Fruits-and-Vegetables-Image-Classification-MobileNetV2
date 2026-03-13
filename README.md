# 🍎🥦 Klasifikasi Buah dan Sayuran Segar & Busuk Menggunakan MobileNetV2

<div align="center">

![Classification Banner](https://img.shields.io/badge/Deep%20Learning-Deteksi%20Kesegaran-FF6B6B?style=for-the-badge&logo=tensorflow&logoColor=white)

**Klasifikasi Kualitas Buah & Sayuran Berbasis AI: Segar atau Busuk?**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.13-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)
[![Akurasi](https://img.shields.io/badge/Akurasi%20Test-95.63%25-success?style=for-the-badge)](https://github.com/NabilaPutriAsySyifa/Fruits-and-Vegetables-Image-Classification-MobileNetV2)

[Tentang Proyek](#-tentang-proyek) • [Problem Statement](#-problem-statement) • [Dataset](#-dataset) • [Arsitektur Model](#️-arsitektur-model) • [Hasil Evaluasi](#-hasil-evaluasi) • [Demo & Inferensi](#-inferensi--demo) • [Cara Menggunakan](#-cara-menggunakan)

</div>

---

## 📊 Tentang Proyek

Proyek **deep learning** untuk klasifikasi otomatis **kesegaran (freshness quality)** buah dan sayuran menggunakan **MobileNetV2** dengan teknik **Transfer Learning** dan **Fine-Tuning**. Model ini mampu membedakan antara produk **segar (fresh)** dan **busuk (rotten)** dari **10 jenis** buah dan sayuran berbeda, dengan total **20 kelas** klasifikasi dan akurasi test mencapai **95,63%**.

### 🎯 Tujuan Proyek

Membangun sistem **deteksi kualitas** yang dapat:
- ✅ Mengklasifikasikan **20 kondisi** produk (Segar + Busuk × 10 jenis)
- ✅ Mencapai akurasi **≥95%** menggunakan custom callback target
- ✅ Memberikan **skor kepercayaan** (confidence score) untuk setiap prediksi
- ✅ Diekspor dalam berbagai format untuk **deployment fleksibel**

**Potensi Penerapan:**
- 🏪 **Quality Control** di gudang & pusat distribusi pangan
- 🛒 **Smart Retail** untuk pengecekan kesegaran produk otomatis
- 📦 **Supply Chain** pemantauan kualitas produk secara real-time
- 🌾 **Pasca Panen** sortasi otomatis berdasarkan kondisi kesegaran

---

## 💼 Problem Statement

### Permasalahan Utama

> **Bagaimana cara mengotomasi proses inspeksi kualitas buah dan sayuran untuk membedakan produk segar vs busuk secara akurat dan efisien?**

### Tantangan yang Dihadapi

**Inspeksi Manual Konvensional:**
- ⏱️ **Memakan Waktu**: Inspektor harus memeriksa ribuan produk setiap harinya
- 👁️ **Subjektif**: Standar "segar" vs "busuk" bisa berbeda-beda antar inspektor
- 💸 **Mahal**: Biaya tenaga kerja tinggi untuk staf quality control
- 📉 **Rentan Kesalahan**: Kelelahan manusia menyebabkan penilaian yang tidak konsisten
- 🚫 **Tidak Skalabel**: Tidak feasible untuk volume produk yang besar

**Keunggulan Solusi AI:**
- ✅ **Deteksi Instan**: Kurang dari 1 detik per produk
- ✅ **Standar Objektif**: Konsisten berdasarkan fitur visual gambar
- ✅ **Hemat Biaya**: Mengurangi biaya tenaga kerja secara signifikan
- ✅ **Skalabel**: Mampu memproses ribuan produk per jam
- ✅ **Operasi 24/7**: Tidak mengenal kelelahan, bekerja terus-menerus

---

## 📁 Dataset

**Sumber**: [Fruits and Vegetables Dataset — Kaggle](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)

Dataset berkualitas tinggi yang dirancang khusus untuk klasifikasi **segar vs busuk** pada buah dan sayuran.

### Komposisi Dataset

```
📦 Total Dataset: 12.000 gambar
├── 🍎 Kategori Buah:    5.997 gambar
├── 🥦 Kategori Sayuran: 6.003 gambar
│
└── Pembagian Dataset (rasio 80:10:10):
    ├── Training   : ~9.600 gambar  (80%)
    ├── Validation : ~1.200 gambar  (10%)
    └── Test       : ~1.200 gambar  (10%)
```

### 20 Kelas (Varian Segar + Busuk)

Dataset terdiri dari **10 jenis produk**, masing-masing dalam **2 kondisi** (Segar & Busuk):

#### 🍎 Buah-buahan (5 jenis × 2 kondisi = 10 kelas)

| Produk | Kelas Segar | Kelas Busuk | Rata-rata Gambar/Kelas |
|--------|-------------|-------------|------------------------|
| 🍎 Apel | Fresh Apple | Rotten Apple | ~600 |
| 🍌 Pisang | Fresh Banana | Rotten Banana | ~600 |
| 🥭 Mangga | Fresh Mango | Rotten Mango | ~600 |
| 🍊 Jeruk | Fresh Orange | Rotten Orange | ~600 |
| 🍓 Stroberi | Fresh Strawberry | Rotten Strawberry | ~600 |

#### 🥦 Sayuran (5 jenis × 2 kondisi = 10 kelas)

| Produk | Kelas Segar | Kelas Busuk | Rata-rata Gambar/Kelas |
|--------|-------------|-------------|------------------------|
| 🫑 Paprika | Fresh Bell Pepper | Rotten Bell Pepper | ~600 |
| 🥕 Wortel | Fresh Carrot | Rotten Carrot | ~600 |
| 🥒 Timun | Fresh Cucumber | Rotten Cucumber | ~600 |
| 🥔 Kentang | Fresh Potato | Rotten Potato | ~600 |
| 🍅 Tomat | Fresh Tomato | Rotten Tomato | ~600 |

### Distribusi Kelas

**Dataset Seimbang** — setiap kelas memiliki **±600 gambar**, memastikan model tidak bias ke kelas tertentu.

```
Pemetaan Kelas (20 kelas, diurutkan abjad):
├── 0:  FreshApple          ├── 10: RottenApple
├── 1:  FreshBanana         ├── 11: RottenBanana
├── 2:  FreshBellpepper     ├── 12: RottenBellpepper
├── 3:  FreshCarrot         ├── 13: RottenCarrot
├── 4:  FreshCucumber       ├── 14: RottenCucumber
├── 5:  FreshMango          ├── 15: RottenMango
├── 6:  FreshOrange         ├── 16: RottenOrange
├── 7:  FreshPotato         ├── 17: RottenPotato
├── 8:  FreshStrawberry     ├── 18: RottenStrawberry
└── 9:  FreshTomato         └── 19: RottenTomato
```

---

## 🔬 Metodologi

### Pendekatan Transfer Learning + Fine-Tuning

**Model Dasar**: MobileNetV2 (pre-trained pada ImageNet)

**Mengapa MobileNetV2?**
- ✅ **Ringan**: Cocok untuk deployment di perangkat mobile/edge
- ✅ **Efisien**: Inferensi cepat untuk deteksi real-time
- ✅ **Terbukti Akurat**: Performa tinggi pada tugas klasifikasi gambar
- ✅ **Mobile-Ready**: Dapat dikonversi ke TFLite untuk perangkat mobile

### Strategi Pelatihan: Fine-Tuning

Model dibangun dengan strategi **fine-tuning bertahap**:

1. **Memuat Base Model MobileNetV2** dengan bobot ImageNet (`include_top=False`)
2. **Fine-Tuning Selektif**: `base_model.trainable = True`, kemudian 100 layer pertama dibekukan (`fine_tune_at = 100`)
3. **Penambahan Custom Head**: Conv2D → MaxPooling2D → GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(20, Softmax)
4. **Pelatihan** dengan optimizer Adam (lr=0,00001) dan callback cerdas

---

## 🏗️ Arsitektur Model

### Struktur Arsitektur

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ mobilenetv2_1.00_224            │ (None, 7, 7, 1280)     │     2.257.984 │
│ (Functional)                    │                        │  [FINE-TUNED] │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 7, 7, 64)       │       737.344 │
│                                 │ [ReLU, padding=same]   │  [TRAINABLE]  │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 3, 3, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 64)             │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │        16.640 │
│                                 │ [ReLU activation]      │  [TRAINABLE]  │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 256)            │             0 │
│                                 │ [rate: 0.5]            │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 20)             │         5.140 │
│                                 │ [Softmax activation]   │  [TRAINABLE]  │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

### Statistik Model

```
📊 Parameter Model:
├── Total params        : 3.017.108 (11,51 MB)
├── Trainable params    : 2.620.564 (10,00 MB)  ← Layer fine-tuned + custom head
└── Non-trainable params:   396.544 ( 1,51 MB)  ← Layer MobileNetV2 yang dibekukan
```

**Penjelasan Setiap Layer:**
- **MobileNetV2 Base**: Mengekstraksi fitur visual (warna, tekstur, bentuk)
- **Conv2D (64 filter)**: Menangkap fitur spasial tambahan dari output MobileNetV2
- **MaxPooling2D**: Mereduksi dimensi spasial
- **GlobalAveragePooling2D**: Meratakan peta fitur menjadi vektor 1D
- **Dense (256)**: Mempelajari pola tingkat tinggi (indikator segar vs busuk)
- **Dropout (0,5)**: Mencegah overfitting
- **Output Dense (20)**: Probabilitas untuk masing-masing dari 20 kelas

---

### Konfigurasi Pelatihan

```python
Optimizer : Adam
    - Learning Rate : 0,00001 (1e-5)

Loss Function : Categorical Crossentropy
    - Sesuai untuk klasifikasi multi-kelas

Augmentasi Data (Training):
    - rescale          : 1./255       # Normalisasi piksel ke [0, 1]
    - rotation_range   : 40°          # Rotasi acak
    - width_shift_range: 0.2          # Geser horizontal acak
    - height_shift_range: 0.2         # Geser vertikal acak
    - shear_range      : 0.2          # Transformasi shear
    - zoom_range       : 0.2          # Zoom acak
    - brightness_range : [0.8, 1.2]   # Variasi kecerahan
    - horizontal_flip  : True         # Cermin horizontal

Callbacks yang Digunakan:
    ✅ TargetAccuracy  : Hentikan training jika train & val accuracy ≥ 95%
    ✅ EarlyStopping   : Hentikan jika val_loss tidak membaik selama 5 epoch
    ✅ ModelCheckpoint : Simpan model terbaik berdasarkan val_accuracy
    ✅ ReduceLROnPlateau: Kurangi LR jika val_loss stagnan (factor=0.2, patience=3)

Class Weights       : Balanced (mengatasi ketidakseimbangan kelas minor)
Batch Size          : 32
Input Size          : 224 × 224 × 3
Maksimum Epoch      : 50
```

---

## 📊 Hasil Evaluasi

### Performa Pelatihan

Model dilatih hingga mencapai **target akurasi 95%** secara serentak pada data training dan validation, sesuai custom callback `TargetAccuracy` yang dikonfigurasi.

**Metrik Pelatihan Akhir (Epoch 49):**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 PELATIHAN SELESAI — Target Akurasi Tercapai!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Akurasi Training    : 95,13% ✅
Akurasi Validasi    : 95,69% ✅

Loss Training       : 0,1674
Loss Validasi       : 0,1537

Epoch Terakhir      : 49 dari 50
Langkah per Epoch   : 299
Learning Rate       : 1,0e-05
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Pelatihan **berhenti lebih awal di Epoch 49** karena akurasi training (95,13%) dan validasi (95,69%) secara bersamaan melampaui ambang batas target ≥95% yang dikonfigurasi melalui custom callback `TargetAccuracy`.

---

### Performa pada Data Test

**Hasil Evaluasi Akhir pada Data Test:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 HASIL EVALUASI DATA TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Akurasi Test  : 95,63%
   Loss Test     : 0,1613

Langkah evaluasi : 38 steps (batch size 32)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Ringkasan Performa Model:**

| Metrik | Nilai | Status |
|--------|-------|--------|
| **Akurasi Test** | **95,63%** | ✅ Melampaui target (≥95%) |
| **Loss Test** | 0,1613 | ✅ Rendah dan stabil |
| **Akurasi Training** | 95,13% | ✅ Tercapai di Epoch 49 |
| **Akurasi Validasi** | 95,69% | ✅ Tidak overfitting |
| **Ukuran Model** | 11,51 MB | ✅ Ringan & efisien |

---

### Visualisasi Kurva Pelatihan

Berdasarkan grafik training history dari notebook:

**Kurva Akurasi:**
- ✅ Akurasi training meningkat bertahap dari ~17% → 95%
- ✅ Akurasi validasi mengikuti dengan selisih kecil (tidak overfitting)
- ✅ Konvergensi stabil di sekitar epoch 45–49

**Kurva Loss:**
- ✅ Loss training turun drastis dari 2,78 → 0,16
- ✅ Loss validasi turun dari 2,02 → 0,15
- ✅ Tidak terjadi divergensi (model berhasil generalisasi dengan baik)

**Observasi Penting:**
- 🎯 **Tidak ada overfitting**: Akurasi validasi mendekati akurasi training sepanjang pelatihan
- 🎯 **Konvergensi mulus**: Tidak ada osilasi atau instabilitas yang signifikan
- 🎯 **Custom callback bekerja**: Training berhenti di titik optimal Epoch 49

---

## 🔍 Inferensi & Demo

### Cara Mencoba Model Langsung

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/NabilaPutriAsySyifa/Fruits-and-Vegetables-Image-Classification-MobileNetV2/blob/main/Submission_Dicoding_Deep_Learning_Akhir_MobileNetV2.ipynb)

Buka notebook di Google Colab, jalankan seluruh cell, lalu upload gambar buah atau sayuran untuk mendapatkan prediksi beserta tingkat kepercayaan model.

### Contoh Output Prediksi

```python
# Upload gambar → model menganalisis → hasil prediksi ditampilkan

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HASIL PREDIKSI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gambar ini terdeteksi sebagai : ROTTEN STRAWBERRY
Tingkat Kepercayaan Model     : 99.95%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Model menampilkan **nama kelas** lengkap beserta **tingkat kepercayaan** dalam persentase, dan memvisualisasikan gambar dengan judul prediksi menggunakan matplotlib.

---

## 🚀 Cara Menggunakan

### 1. Setup Environment

```bash
# Clone repositori
git clone https://github.com/NabilaPutriAsySyifa/Fruits-and-Vegetables-Image-Classification-MobileNetV2.git
cd Fruits-and-Vegetables-Image-Classification-MobileNetV2

# Install dependensi
pip install tensorflow keras numpy matplotlib seaborn Pillow scikit-learn split-folders tensorflowjs
```

### 2. Unduh Dataset

**Opsi A: Kaggle CLI (Direkomendasikan)**
```bash
pip install kaggle
kaggle datasets download -d muhriddinmuxiddinov/fruits-and-vegetables-dataset
unzip fruits-and-vegetables-dataset.zip
```

**Opsi B: Unduh Manual**
1. Kunjungi [halaman dataset Kaggle](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)
2. Klik tombol "Download" (membutuhkan akun Kaggle)
3. Ekstrak ke folder proyek

### 3. Pelatihan Model

**Rekomendasi: Google Colab dengan GPU T4**

```python
# 1. Upload notebook ke Google Colab
# 2. Upload file kaggle.json untuk autentikasi API
# 3. Jalankan cell secara berurutan dari atas ke bawah
# 4. Model terbaik akan otomatis tersimpan di:
#    - mobilenetv2_fruits_veg_best.keras  (checkpoint terbaik)
#    - saved_model/                        (TensorFlow SavedModel)
#    - tfjs_model/                         (TensorFlow.js)
#    - model.tflite                        (TensorFlow Lite)
#    - label.txt                           (daftar label kelas)
```

### 4. Load Model & Prediksi

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Muat SavedModel
loaded_model = tf.saved_model.load('saved_model')
infer = loaded_model.signatures["serving_default"]

# Daftar nama kelas (20 kelas, urutan abjad)
class_names = [
    'FreshApple', 'FreshBanana', 'FreshBellpepper', 'FreshCarrot',
    'FreshCucumber', 'FreshMango', 'FreshOrange', 'FreshPotato',
    'FreshStrawberry', 'FreshTomato', 'RottenApple', 'RottenBanana',
    'RottenBellpepper', 'RottenCarrot', 'RottenCucumber', 'RottenMango',
    'RottenOrange', 'RottenPotato', 'RottenStrawberry', 'RottenTomato'
]

def prediksi_kesegaran(image_path):
    # Muat dan proses gambar sesuai standar MobileNetV2
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi menggunakan SavedModel
    predictions = infer(tf.constant(img_array))
    output_key = list(predictions.keys())[0]
    predicted_index = np.argmax(predictions[output_key].numpy(), axis=1)[0]
    confidence = np.max(predictions[output_key].numpy(), axis=1)[0]

    predicted_class = class_names[predicted_index]
    kondisi = "SEGAR ✅" if "Fresh" in predicted_class else "BUSUK ❌"

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" HASIL PREDIKSI")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Gambar terdeteksi sebagai : {predicted_class.upper()}")
    print(f"Kondisi                   : {kondisi}")
    print(f"Tingkat Kepercayaan       : {confidence * 100:.2f}%")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Contoh penggunaan
prediksi_kesegaran('gambar_buah_sayur.jpg')
```

**Contoh Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HASIL PREDIKSI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gambar terdeteksi sebagai : ROTTEN STRAWBERRY
Kondisi                   : BUSUK ❌
Tingkat Kepercayaan       : 99.95%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 💡 Wawasan & Pembelajaran

### Faktor Keberhasilan Model

**1. Kekuatan Transfer Learning + Fine-Tuning**
- ✅ Memanfaatkan pengetahuan ImageNet (1,4 juta gambar) sebagai fondasi
- ✅ Fine-tuning selektif: 100 layer pertama dibekukan, sisanya dilatih ulang
- ✅ Model hanya perlu belajar fitur spesifik "segar vs busuk" di atas fitur umum yang sudah ada

**2. Kualitas Data**
- ✅ Dataset seimbang (±600 gambar per kelas) → tidak ada bias kelas
- ✅ Perbedaan visual yang jelas antara kondisi segar dan busuk
- ✅ Augmentasi data yang kaya → model lebih robust terhadap variasi gambar nyata

**3. Desain Arsitektur**
- ✅ MobileNetV2 sebagai ekstraktor fitur yang efisien
- ✅ Conv2D tambahan untuk menangkap fitur spasial lebih dalam
- ✅ Dropout 0,5 untuk mencegah overfitting secara efektif

**4. Strategi Pelatihan**
- ✅ Custom callback `TargetAccuracy` untuk berhenti tepat di titik optimal
- ✅ `ReduceLROnPlateau` untuk penyesuaian learning rate otomatis
- ✅ Class weights seimbang untuk menangani distribusi yang sedikit tidak merata

---

### Rencana Pengembangan ke Depan

**Peningkatan Model:**
- [ ] Implementasi **Grad-CAM** untuk memvisualisasikan area keputusan model
- [ ] Tambahkan **klasifikasi tingkat kerusakan** (awal / sedang / parah)
- [ ] Bandingkan performa dengan **EfficientNet** atau **ResNet50**
- [ ] Ensemble beberapa model untuk meningkatkan keandalan

**Penambahan Fitur:**
- [ ] **Multi-object detection** (deteksi beberapa item dalam satu gambar)
- [ ] **Estimasi persentase kesegaran** (skor 0–100%)
- [ ] **Prediksi umur simpan** berdasarkan indikator visual

**Deployment:**
- [ ] Build **REST API** menggunakan FastAPI atau Flask
- [ ] Buat **web app demo** dengan Streamlit atau Gradio
- [ ] **Edge deployment** (Raspberry Pi + kamera) menggunakan model TFLite

---

## 🛠️ Teknologi yang Digunakan

| Keperluan | Teknologi | Versi |
|-----------|-----------|-------|
| **Deep Learning** | TensorFlow / Keras | 2.19 / 3.13 |
| **Model Dasar** | MobileNetV2 | Pre-trained ImageNet |
| **Pemrosesan Gambar** | PIL / Pillow | Latest |
| **Komputasi Numerik** | NumPy | Latest |
| **Visualisasi** | Matplotlib & Seaborn | Latest |
| **Class Weights** | scikit-learn | Latest |
| **Pembagian Dataset** | split-folders | 0.6.1 |
| **Ekspor Web** | TensorFlow.js | 4.22.0 |
| **Ekspor Mobile** | TensorFlow Lite | Bawaan TF |
| **Lingkungan Pelatihan** | Google Colab + GPU T4 | Free Tier |
| **Sumber Dataset** | Kaggle | 12.000 gambar |

---

## 📂 Struktur Proyek

```
Fruits-and-Vegetables-Image-Classification-MobileNetV2/
│
├── 📓 Submission_Dicoding_Deep_Learning_Akhir_MobileNetV2.ipynb  # Notebook utama
├── 🐍 submission_dicoding_deep_learning_akhir_mobilenetv2.py     # Versi skrip Python
│
├── 📁 saved_model/                   # Model format TensorFlow SavedModel
│   ├── assets/
│   ├── variables/
│   └── saved_model.pb
│
├── 📁 tfjs_model/                    # Model format TensorFlow.js
│   ├── model.json
│   └── group1-shard*.bin
│
├── 🏆 mobilenetv2_fruits_veg_best.keras   # Checkpoint model terbaik
├── 📱 model.tflite                        # Model TensorFlow Lite (mobile)
├── 🏷️  label.txt                          # Daftar label 20 kelas
└── 📄 README.md
```

---

## 👩‍💻 Tentang Pembuat

<div align="center">

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/NabilaPutriAsySyifa.png" width="100px;" alt="Nabila Putri Asy Syifa"/><br />
      <sub><b>Nabila Putri Asy Syifa</b></sub><br />
      <sub>👩🏻‍💻 Deep Learning Engineer</sub><br />
      <sub>Model Development · Training · Deployment</sub>
    </td>
  </tr>
</table>

**Proyek Deep Learning Pribadi — 2025**

*Membangun AI untuk otomasi deteksi kualitas pangan*

</div>

---

## 📞 Kontak

- 📧 Email: nabilaputriasysyifa99@gmail.com
- 🐙 GitHub: [@NabilaPutriAsySyifa](https://github.com/NabilaPutriAsySyifa)
- 📊 Portfolio: [Lihat Proyek Lainnya](https://github.com/NabilaPutriAsySyifa?tab=repositories)

---

## 🙏 Ucapan Terima Kasih

Terima kasih kepada:
- **Kaggle** atas penyediaan dataset berkualitas tinggi secara terbuka
- **Google Colab** atas akses GPU gratis yang memungkinkan pelatihan model
- **Tim TensorFlow & Keras** atas framework deep learning yang luar biasa
- **Para Penulis MobileNetV2** (Sandler et al., 2018) atas arsitektur yang efisien
- **Komunitas Open Source** atas tools dan resources yang digunakan

---

## 📚 Referensi

**Paper Utama:**
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018

**Dataset:**
- [Fruits and Vegetables Dataset](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset) — Kaggle (CC0-1.0)

**Sumber Teknis:**
- [Panduan Transfer Learning TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras Applications — MobileNetV2](https://keras.io/api/applications/mobilenet/)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan **portofolio dan pembelajaran**.

**Lisensi Dataset:** CC0-1.0 (Domain Publik) — lihat [halaman dataset Kaggle](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)

**Lisensi Kode:** MIT License — bebas digunakan untuk pembelajaran & keperluan non-komersial.

---

<div align="center">

**Proyek Deteksi Kualitas Pangan Berbasis AI**

*Dari 12.000 gambar mentah menuju klasifikasi 20 kelas dengan akurasi 95,63%*

⭐ Beri bintang pada repositori ini jika proyeknya bermanfaat bagimu!

---

© 2025 Nabila Putri Asy Syifa | Deep Learning Portfolio

**[📓 Lihat Notebook](https://github.com/NabilaPutriAsySyifa/Fruits-and-Vegetables-Image-Classification-MobileNetV2/blob/main/Submission_Dicoding_Deep_Learning_Akhir_MobileNetV2.ipynb)** | **[📊 Dataset](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)** | **[🚀 Live Demo](https://colab.research.google.com/github/NabilaPutriAsySyifa/Fruits-and-Vegetables-Image-Classification-MobileNetV2/blob/main/Submission_Dicoding_Deep_Learning_Akhir_MobileNetV2.ipynb)**

</div>
