# Laporan Proyek Machine Learning - Adam Firmansyah

## Domain Proyek

Penyakit kardiovaskular adalah gangguan pada jantung yang terjadi akibat kurangnya pasokan darah ke otot jantung. Di Indonesia, penyakit ini menjadi penyebab kematian tertinggi kedua setelah stroke, dengan kontribusi yang cukup besar terhadap total kematian. Terdapat hubungan yang erat antara obesitas dan kejadian penyakit kardiovaskular, seperti yang terlihat di rumah sakit umum tertentu. Obesitas dapat memicu peningkatan tekanan darah (hipertensi), kadar trigliserida dan kolesterol, resistensi glukosa, serta risiko terjadinya penggumpalan darah—semuanya merupakan faktor risiko utama penyakit kardiovaskular.

Obesitas sendiri merupakan kondisi kelebihan lemak pada jaringan adiposa dan termasuk salah satu masalah kesehatan yang umum, namun sebenarnya dapat dicegah, misalnya dengan meningkatkan aktivitas fisik. Gaya hidup juga sangat berpengaruh terhadap risiko terjadinya obesitas dan penyakit kardiovaskular. Oleh karena itu, observasi dalam penelitian ini menggunakan sejumlah variabel, yaitu usia (dalam hari), jenis kelamin, tinggi badan, berat badan, tekanan darah, kadar kolesterol, kadar glukosa, kebiasaan merokok, konsumsi alkohol, tingkat aktivitas fisik, dan status apakah seseorang menderita penyakit kardiovaskular atau tidak.

## Business Understanding

### Problem Statements

Rumusan masalah dari latar belakang diatas adalah:

1. Seberapa akurat model machine learning dapat memprediksi risiko penyakit kardiovaskular berdasarkan fitur-fitur kesehatan individu seperti tekanan darah, kolesterol, dan glukosa?

2. Apakah jenis kelamin dan kelompok usia tertentu memiliki risiko lebih tinggi terhadap penyakit kardiovaskular?

### Goals

Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini.

1. Menganalisis seberapa akurat model machine learning dalam memprediksi risiko penyakit kardiovaskular berdasarkan variabel-variabel kesehatan seperti tekanan darah, kadar kolesterol, kadar glukosa, serta faktor gaya hidup.

2. Mengidentifikasi hubungan antara jenis kelamin dan kelompok usia dengan kemungkinan menderita penyakit kardiovaskular, guna mengetahui apakah kelompok tertentu memiliki risiko yang lebih tinggi.

### Solution statements

1. Membangun model prediktif berbasis machine learning, seperti Logistic Regression, Random Forest, atau XGBoost, untuk mengklasifikasikan apakah seseorang berisiko terkena penyakit kardiovaskular berdasarkan data yang tersedia dalam dataset.

2. Melakukan analisis eksploratif dan statistik terhadap data usia dan jenis kelamin untuk mengetahui apakah terdapat pola atau kecenderungan risiko lebih tinggi pada kelompok usia atau gender tertentu.

3. Mengukur performa model dengan metrik evaluasi seperti akurasi, precision, recall, dan F1-score, serta menggunakan confusion matrix untuk menilai efektivitas klasifikasi model.

### Metodologi

Prediksi penyakit cardiovascular adalah tujuan yang ingin dicapai. berdasarkan penelitian yang ada. Dalam predictive metodologi pada proyek ini adalah membangun model klasifikasi dengan cardiovascular sebagai target.

### Metrik

Metrik yang digunakan untuk mengevaluasi seberapa baik model klasifikasi merupakan confusion matrix.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Cardiovascular Disease Dataset, yang diambil dari platform open-source Kaggle dan dipublikasikan oleh Dina Sulianova. Dataset ini mencakup data kesehatan pasien dari tahun 2007 hingga 2010, yang dikumpulkan oleh Pusat Informasi Medis di Rusia.

Dataset ini bertujuan untuk membantu dalam prediksi risiko penyakit kardiovaskular menggunakan data seperti tekanan darah, kadar kolesterol, glukosa, serta faktor gaya hidup. Total terdapat 70.000 entri data pasien dengan 13 variabel, termasuk variabel target cardio.

Informasi Dataset

- Jumlah entri: 70.000

- Jumlah fitur: 13 (termasuk variabel target)

- Variabel target: cardio (0 = tidak memiliki penyakit kardiovaskular, 1 = memiliki)

### Variabel-variabel yang memengaruhi risiko penyakit kardiovaskular adalah sebagai berikut:

| Variabel    | Keterangan                                                           | Contoh Nilai |
| ----------- | -------------------------------------------------------------------- | ------------ |
| id          | ID unik untuk masing-masing pasien                                   | 12345        |
| age         | Usia pasien dalam satuan **hari**                                    | 18393        |
| gender      | Jenis kelamin pasien (1 = Wanita, 2 = Pria)                          | 2            |
| height      | Tinggi badan dalam **sentimeter**                                    | 168          |
| weight      | Berat badan dalam **kilogram**                                       | 70           |
| ap_hi       | Tekanan darah **sistolik** (atas)                                    | 120          |
| ap_lo       | Tekanan darah **diastolik** (bawah)                                  | 80           |
| cholesterol | Kadar kolesterol (1 = normal, 2 = di atas normal, 3 = sangat tinggi) | 2            |
| gluc        | Kadar glukosa (1 = normal, 2 = di atas normal, 3 = sangat tinggi)    | 1            |
| smoke       | Apakah pasien **merokok**? (0 = tidak, 1 = ya)                       | 0            |
| alco        | Apakah pasien **mengonsumsi alkohol**? (0 = tidak, 1 = ya)           | 1            |
| active      | Apakah pasien **aktif secara fisik**? (0 = tidak, 1 = ya)            | 1            |
| cardio      | **Target klasifikasi**: memiliki penyakit kardiovaskular atau tidak  | 0 / 1        |

Catatan Penting:

- Kolom age dikonversi menjadi usia dalam tahun untuk memudahkan interpretasi.

- Kombinasi fitur height dan weight dapat digunakan untuk menghitung BMI (Body Mass Index).

- Data mengandung fitur biner dan kategorikal yang telah dinormalisasi dalam proses preprocessing.

### Exploratory Data Analysis - Univariate Analysis

Melakukan Univariate Analysis untuk numerical dan categorical features.

- Numerical Features pada dataset ini terdiri atas: ['umur', 'height', 'weight', 'ap_hi', 'ap_lo']

- Categorical Feature pada dataset ini terdiri atas ['gender','cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']

<p align="center">
  <img src="https://github.com/user-attachments/assets/79426f7f-03e3-460c-aaa6-2ed8e34355c7" width="1000"/>
</p>

Gambar di atas dapat diinterpretasikan sebagai berikut.

1. Distribusi Jenis Kelamin
   Laki-laki (label 2) jauh lebih banyak dibandingkan perempuan (label 1). Hal ini menunjukkan bahwa populasi dalam dataset didominasi oleh pasien laki-laki.

2. Kadar Kolesterol
   Sebagian besar pasien memiliki kadar kolesterol normal (label 1). Namun, ada proporsi signifikan yang memiliki kolesterol di atas normal (label 2) dan jauh di atas normal (label 3), yang menjadi indikator risiko penting terhadap penyakit kardiovaskular.

3. Kadar Glukosa
   Mayoritas pasien memiliki kadar glukosa normal (label 1). Tetapi tetap ada kelompok yang memiliki kadar glukosa tinggi, yang bisa dikaitkan dengan diabetes, salah satu faktor risiko cardiovascular.

4. Merokok (smoke)
   Sebagian besar pasien tidak merokok (label 0). Namun, ada minoritas yang merokok, dan data ini tetap penting karena merokok meningkatkan risiko cardiovascular.

5. Konsumsi Alkohol (alco)
   Mayoritas pasien tidak mengonsumsi alkohol. Konsumsi alkohol relatif rendah dalam dataset, namun tetap relevan sebagai faktor gaya hidup yang berkontribusi terhadap risiko.

6. Aktivitas Fisik (active)
   Sebagian besar pasien terdata aktif secara fisik (label 1). Artinya ada kesadaran olahraga atau aktivitas di antara responden, meskipun tetap ditemukan penderita cardiovascular di kelompok ini.

7. Penyakit Kardiovaskular (cardio)
   Dataset relatif seimbang antara pasien yang menderita penyakit kardiovaskular (label 1) dan yang tidak (label 0). Ini bagus untuk keperluan klasifikasi model machine learning karena tidak terlalu imbalanced.

### Exploratory Data Analysis - Multivariate Analysis

Melakukan Multivariate Analysis untuk menganalisis hubungan antar variabel

#### 1. Mengetahui distribusi Umur terhadap Risiko Penyakit

<p align="center">
  <img src="https://github.com/user-attachments/assets/32db376d-11b8-46e1-a06e-ac74cb6056d3" width="600"/>
</p>

Insight:

- Penyakit kardiovaskular lebih banyak terjadi pada usia di atas 50 tahun.

- Usia adalah prediktor penting dalam risiko penyakit ini.

#### 2. Mengetahui distribusi Tekanan Darah Terhadap Risiko Penyakit (Cardio)

<p align="center">
  <img src="https://github.com/user-attachments/assets/29de156f-7c67-45b7-9f3d-1c9cb9a4fa81" width="600"/>
</p>

Insight:
Pasien dengan penyakit kardiovaskular cenderung memiliki tekanan darah yang lebih tinggi dibandingkan yang tidak.

#### 3. Mengetahui BMI (Body Mass Index) vs Risiko Penyakit

<p align="center">
  <img src="https://github.com/user-attachments/assets/d6a5f287-0383-4524-bf2c-821c397ed7af" width="1000"/>
</p>

Insight:

- BMI lebih tinggi tampak lebih sering pada pasien dengan penyakit cardiovascular.

- Obesitas merupakan faktor risiko yang nyata.

#### 4. Mengetahui fitur Lifestyle (Smoke, Alcohol, Active) terhadap Risiko

<p align="center">
  <img src="https://github.com/user-attachments/assets/a485f71f-b84d-4e1d-82b1-ad7288b581ba" width="1000"/>
</p>

Insight:

- Pasien yang tidak aktif berolahraga cenderung memiliki lebih banyak kasus penyakit cardiovascular.

- Pola untuk merokok dan konsumsi alkohol tidak terlalu dominan, mungkin karena datanya imbalanced.

### Data Quality Verification

#### Memeriksa data duplikat

<p align="center">
  <img src="https://github.com/user-attachments/assets/80d4de62-3c92-4139-8d7d-815ed01e3b55" width="300"/>
</p>
> Berdasarkan program di atas, maka diketahui tidak terdapat data duplikat. Oleh karena itu tidak perlu di drop data pada bagian data cleaning

#### Memeriksa data missing value

<p align="center">
  <img src="https://github.com/user-attachments/assets/f9307676-d648-431d-9aa8-2eafe4eb8d05" width="300"/>
</p>
> Berdasarkan program di atas, maka diketahui tidak terdapat missing values. Oleh karena itu tidak perlu di drop data pada bagian data cleaning

#### Memeriksa Outlier

<p align="center">
  <img src="https://github.com/user-attachments/assets/07f83da4-0f0a-4cc1-af7b-23be480689d4" width="300"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/bd498cba-e7e3-44b3-9b9d-fc665dd67ac7" width="300"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/094c4d91-2a65-4dbb-8e97-8f9a3cce9c8b" width="300"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/c5802446-1461-4cbb-8210-92f12c94f883" width="300"/>
</p>

Insight :

1. Pada variabel Umur, terlihat bahwa sebagian besar responden berada dalam rentang usia 48 hingga 58 tahun. Ditemukan dua data pencilan (outlier) pada usia sekitar 30 tahun ke atas. Namun, data ini tetap dipertahankan karena secara logis masih memungkinkan seseorang berusia di atas 30 tahun mengalami kondisi yang diteliti.

2. Untuk variabel Weight, mayoritas responden memiliki berat badan antara 60 hingga 80 kilogram. Terdapat sejumlah outlier, terutama pada rentang berat 20 hingga 35 kg. Karena nilai tersebut tergolong tidak wajar untuk individu berusia di atas 35 tahun, maka data pencilan ini akan ditindaklanjuti.

3. Pada variabel Height, sebagian besar responden memiliki tinggi badan antara 1,6 hingga 1,7 meter. Meskipun terdapat banyak outlier, data dengan tinggi badan ekstrem seperti 0,5 hingga 1 meter dianggap tidak realistis untuk usia di atas 30 tahun, sehingga akan ditangani lebih lanjut.

4. Pada variabel ap_hi, ditemukan cukup banyak outlier. Data dengan nilai tekanan darah melebihi 500 mmHg atau bernilai negatif dinilai tidak masuk akal secara medis dan akan dihapus dari analisis.

5. Demikian pula, variabel ap_lo menunjukkan adanya outlier dengan nilai ekstrem yang tidak logis secara medis, seperti lebih dari 500 mmHg atau nilai negatif. Oleh karena itu, data tersebut juga akan dihapus.

6. Terdapat dua outlier di sisi kiri (sekitar usia 29–30 tahun). Meskipun demikian, usia ini masih tergolong realistis, terutama jika penelitian ini tidak membatasi usia minimum responden.

7. Terdapat jumlah outlier yang sangat banyak, bahkan mencapai nilai di atas 300, yang secara medis tidak mungkin terjadi. Outlier ekstrem ini menunjukkan adanya anomali data, kemungkinan besar akibat kesalahan input atau pencatatan data.

#### Memeriksa Data yang memiliki nilai 0 pada kolom
<p align="center">
  <img src="https://github.com/user-attachments/assets/1dcb921d-649b-46c7-8522-387ca2d6c152" width="300"/>
</p>
Insight :
terdapat nilai 0 pada kolom ap_lo maka sebanyak 21

## Data Preparation

### Data Cleaning

#### Menghapus Kolom Tidak Relevan
<p align="center">
  <img src="https://github.com/user-attachments/assets/6de8abba-36cf-4f16-9466-4757188c72d1](https://github.com/user-attachments/assets/2154e7e0-48a6-4d39-a380-d1d6b1b9df48" width="1000"/>
</p>
> Program di atas digunakan untuk menghapus kolom yang tidak relevan atau tidak digunakan.

#### Menghapus baris yang terdapat nilai 0 pada masing masing kolom
<p align="center">
  <img src="https://github.com/user-attachments/assets/6de8abba-36cf-4f16-9466-4757188c72d1](https://github.com/user-attachments/assets/2154e7e0-48a6-4d39-a380-d1d6b1b9df48](https://github.com/user-attachments/assets/3f6b806e-1ac1-4d6f-9589-12936a390dca" width="1000"/>
</p>
> Program di atas digunakan untuk menghapus baris yang terdapat nilai 0 pada masing masing kolom.

#### Menangani Outlier
<p align="center">
  <img src="https://github.com/user-attachments/assets/83f6c47e-161a-4e60-8f94-e0df73936caa" width="1000"/>
</p>
> Program di atas digunakan untuk menangani outlier.


### Data Splitting

Target pada dataset ini adalah variabel `cardio` untuk mengetahui akurasi prediksi dari cardio, maka akan menghapus kolom tersebut dari data dan assign kolom tersebut ke variabel baru. Selanjutnya, melakukan split data dengan skema data training sebesar 80% untuk melatih model dan 20% untuk menguji model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1b5c4da3-fdb1-4fc3-9b34-ad7b0e2d3499" width="1000"/>
</p>

### Standarisasi

Standarisasi merupakan proses transformasi data agar setiap fitur memiliki skala yang seragam, umumnya dengan distribusi rata-rata 0 dan standar deviasi 1. Proses ini sangat penting dalam machine learning karena:

- Algoritma seperti Gradient Boosting, Logistic Regression, dan k-NN bekerja lebih optimal ketika input berada pada skala yang sebanding.
- Menghindari dominasi fitur dengan nilai numerik besar terhadap fitur lain yang bernilai kecil.
- Mempercepat proses konvergensi dan meningkatkan stabilitas model selama pelatihan.
- Pada proyek ini, standarisasi diterapkan pada fitur numerik seperti height, weight, ap_hi, ap_lo, umur, dan BMI untuk menyamakan skala antar fitur, sehingga model dapat belajar dari data secara adil dan efisien tanpa bias terhadap fitur tertentu.

#### 1. Scaling untuk data training

<p align="center">
  <img src="https://github.com/user-attachments/assets/ce9684d2-ce7d-43ca-bd0e-77228cc13737" width="1000"/>
</p>

#### 2. Scaling untuk data testing

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d649f8e-4d98-496b-984e-a428740c8c90" width="1000"/>
</p>

## Modeling
---

### 1. Algoritma Logistic Regression

Logistic Regression adalah algoritma supervised learning yang sering digunakan untuk masalah klasifikasi biner. Algoritma ini memodelkan probabilitas kelas target menggunakan fungsi logistik (sigmoid) yang menghasilkan output antara 0 dan 1. Model ini bekerja dengan mencari koefisien yang memaksimalkan likelihood data training.

Parameter yang digunakan:

* **penalty = 'l2'**. Jenis regularisasi untuk mencegah overfitting dengan menambahkan penalti terhadap koefisien model. L2 adalah regularisasi Ridge yang umum digunakan.
* **C = 1.0**. Parameter invers regulasi yang mengatur kekuatan regularisasi. Nilai kecil berarti regularisasi kuat, dan sebaliknya.
* **solver = 'lbfgs'**. Algoritma optimisasi untuk memaksimalkan likelihood, cocok untuk dataset kecil hingga menengah.
* **random\_state = 0**. Menjamin hasil yang konsisten pada pelatihan ulang.

Hasil akurasi model Logistic Regression adalah **67%**.

---

### 2. Algoritma Random Forest

Random Forest adalah algoritma ensemble learning yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting. Model menggunakan agregasi voting dari hasil banyak pohon untuk prediksi klasifikasi.

Parameter yang digunakan:

* **n\_estimators = 100**. Jumlah pohon keputusan yang dibangun. Semakin banyak pohon, model biasanya lebih stabil dan akurat, namun membutuhkan waktu lebih lama.
* **criterion = 'gini'**. Kriteria untuk memilih split terbaik. Gini mengukur impurity data di setiap node.
* **max\_depth = 10**. Batas kedalaman pohon untuk menghindari overfitting.
* **random\_state = 42**. Agar hasil pelatihan konsisten saat pengulangan.
* **n\_jobs = -1**. Menggunakan semua core CPU untuk mempercepat pelatihan.

Hasil akurasi model Random Forest adalah **71%**.

---

### 3. Algoritma K-Nearest Neighbors (KNN)

KNN adalah algoritma berbasis instance yang mengklasifikasikan data baru berdasarkan mayoritas label dari k tetangga terdekatnya di ruang fitur.

Parameter yang digunakan:

* **n\_neighbors = 5**. Jumlah tetangga terdekat yang dipertimbangkan saat klasifikasi.
* **weights = 'uniform'**. Semua tetangga dianggap memiliki bobot yang sama saat voting.
* **metric = 'minkowski'**. Metode pengukuran jarak, dengan p=2 setara dengan jarak Euclidean.
* **algorithm = 'auto'**. Memilih algoritma pencarian tetangga terbaik secara otomatis.

Hasil akurasi model KNN adalah **65%**.

---

### 4. Algoritma Support Vector Machine (SVM)

Support Vector Machine adalah algoritma yang mencari hyperplane optimal untuk memisahkan kelas dengan margin terbesar, mampu menangani kasus linear maupun non-linear menggunakan kernel.

Parameter yang digunakan:

* **kernel = 'rbf'**. Radial Basis Function kernel untuk batas pemisah non-linear yang fleksibel.
* **C = 1.0**. Parameter regularisasi yang mengontrol trade-off antara margin besar dan klasifikasi benar pada training data.
* **gamma = 'scale'**. Parameter kernel yang mengatur jangkauan pengaruh titik data.
* **random\_state = 42**. Menjamin hasil pelatihan yang dapat direproduksi.

Hasil akurasi model SVM adalah **63%**.

---

### 5. Algoritma Decision Tree

Decision Tree adalah model berbasis pohon yang membuat keputusan berdasarkan fitur dengan membagi data di setiap node untuk mencapai klasifikasi yang akurat.

Parameter yang digunakan:

* **criterion = 'gini'**. Kriteria untuk memilih split terbaik.
* **max\_depth = 5**. Membatasi kedalaman pohon agar tidak overfitting.
* **random\_state = 0**. Untuk memastikan hasil yang konsisten.

Hasil akurasi model Decision Tree adalah **53%**.

---

### 6. Algoritma Gradient Boosting

Gradient Boosting adalah algoritma ensemble yang membangun model secara bertahap dengan mengoptimalkan residual dari model sebelumnya menggunakan metode gradient descent.

Parameter yang digunakan:

* **n\_estimators = 100**. Jumlah pohon yang dibuat secara bertahap.
* **learning\_rate = 0.1**. Kecepatan pembelajaran model, semakin kecil semakin lambat pembelajaran tapi biasanya lebih stabil.
* **max\_depth = 3**. Batas kedalaman pohon untuk menghindari overfitting.
* **random\_state = 42**. Menjamin konsistensi hasil pelatihan.

Hasil akurasi model Gradient Boosting adalah **71%**.

---

### 7. Algoritma AdaBoost

AdaBoost (Adaptive Boosting) adalah algoritma ensemble yang menggabungkan banyak model sederhana (biasanya pohon keputusan kecil) dengan memberikan bobot lebih pada data yang sulit diklasifikasikan.

Parameter yang digunakan:

* **n\_estimators = 50**. Jumlah model dasar yang digabungkan.
* **learning\_rate = 1.0**. Mengatur kontribusi masing-masing model.
* **random\_state = 42**. Agar hasil pelatihan konsisten.

Hasil akurasi model AdaBoost adalah **67%**.

---

### 8. Algoritma Naive Bayes

Naive Bayes adalah algoritma probabilistik yang menggunakan teorema Bayes dengan asumsi independensi antar fitur untuk melakukan klasifikasi.

Parameter yang digunakan:

* **variasi model tergantung jenis Naive Bayes, misalnya GaussianNB untuk data kontinu**.
* Algoritma ini tidak memiliki banyak parameter untuk diatur, hanya mempelajari probabilitas dari data training.

Hasil akurasi model Naive Bayes adalah **69%**.

---
### Pemilihan Model Terbaik

<p align="center">
  <img src="https://github.com/user-attachments/assets/54011bed-f248-47ce-8a43-2467e54688e4" width="600"/>
</p>

**📌 Insight:**
Model Gradient Boosting mampu mengklasifikasikan pasien dengan akurasi sekitar 71%, menunjukkan performa yang layak untuk digunakan dalam prediksi risiko awal.

Model ini mengungguli metode klasik seperti Logistic Regression dan Decision Tree, berkat kemampuannya menangkap kompleksitas interaksi antar variabel.

## Evaluation

### 1. Logistic Regression
Model linear yang digunakan untuk klasifikasi biner. Logistic Regression memodelkan probabilitas kejadian menggunakan fungsi sigmoid, dan cocok untuk memprediksi apakah seseorang terkena penyakit kardiovaskular (1) atau tidak (0).
<p align="center">
  <img src="https://github.com/user-attachments/assets/de2ee7ba-72f3-484a-9a1d-64c2076003fd" width="1000"/>
</p>
Insight :

True Positives (TP): 6243 — model berhasil mengenali pasien yang memang terkena penyakit kardiovaskular.

True Negatives (TN): 3018 — model dengan benar mengenali pasien yang tidak terkena penyakit.

False Positives (FP): 3752 — pasien yang sebenarnya tidak terkena, tapi dikira terkena oleh model (bisa menyebabkan kecemasan atau tes lanjutan yang tidak perlu).

False Negatives (FN): 767 — paling krusial, karena pasien yang sebenarnya terkena tidak dikenali oleh model → ini bisa berbahaya dan fatal karena tidak mendapat penanganan tepat waktu.


### 2. Random Forest
Algoritma ensemble yang terdiri dari banyak decision tree. Model ini menggabungkan prediksi dari setiap pohon menggunakan voting mayoritas untuk menghasilkan prediksi akhir. Random Forest efektif menangani data dengan banyak fitur dan interaksi kompleks.
<p align="center">
  <img src="https://github.com/user-attachments/assets/92a82e07-a5d4-4e98-8df8-4246c82c50ff" width="1000"/>
</p>
Insight :

True Positives (TP): 6001
Model berhasil mengenali pasien yang memang terkena penyakit kardiovaskular.

True Negatives (TN): 3823
Model dengan benar mengenali pasien yang tidak terkena penyakit.

False Positives (FP): 2947
Pasien yang sebenarnya tidak terkena, tapi diklasifikasikan sebagai terkena oleh model. Ini bisa menyebabkan kecemasan atau pemeriksaan medis yang tidak perlu.

False Negatives (FN): 1009
Pasien yang sebenarnya terkena tetapi tidak dikenali oleh model. Ini adalah kesalahan paling krusial karena pasien tersebut bisa tidak mendapat penanganan yang tepat waktu.

### 3. K-Nearest Neighbors (K-NN)
Model berbasis instance-based learning yang memprediksi kelas berdasarkan mayoritas tetangga terdekat dari titik data baru. K-NN sensitif terhadap skala data, sehingga standarisasi sangat penting sebelum pelatihan.
<p align="center">
  <img src="https://github.com/user-attachments/assets/1e2007d5-998d-467e-8680-68b358ccfdec" width="1000"/>
</p>
Insight :

True Positives (TP): 4734
Model berhasil mengenali pasien yang memang terkena penyakit kardiovaskular.

True Negatives (TN): 4267
Model dengan benar mengenali pasien yang tidak terkena penyakit.

False Positives (FP): 2503
Pasien yang sebenarnya tidak terkena, tapi diklasifikasikan sebagai terkena oleh model. Hal ini dapat menimbulkan kecemasan dan pemeriksaan medis yang tidak diperlukan.

False Negatives (FN): 2276
Pasien yang sebenarnya terkena tetapi tidak dikenali oleh model. Ini sangat berisiko karena dapat menyebabkan pasien tidak mendapatkan pengobatan yang semestinya.

### 4. Support Vector Machine (SVM)
Model klasifikasi yang mencari hyperplane optimal untuk memisahkan dua kelas data. SVM sangat efektif untuk dataset berdimensi tinggi dan bekerja dengan kernel untuk menangani data non-linear.
<p align="center">
  <img src="https://github.com/user-attachments/assets/f8e5e026-db50-416f-b20e-349fb846c55b" width="1000"/>
</p>
Insight :

True Positives (TP): 5214
Model berhasil mengidentifikasi pasien yang memang terkena penyakit kardiovaskular.

True Negatives (TN): 3543
Model mengenali pasien yang tidak terkena penyakit secara benar.

False Positives (FP): 3227
Pasien yang sebenarnya tidak terkena penyakit namun diklasifikasikan terkena oleh model. Ini bisa menyebabkan pemeriksaan medis tidak perlu.

False Negatives (FN): 1796
Kasus yang seharusnya terdeteksi sebagai terkena namun tidak terdeteksi. Ini sangat berisiko karena bisa mengakibatkan tidak diberikannya penanganan medis yang diperlukan.

### 5. Decision Tree
Algoritma berbasis pohon yang membagi dataset berdasarkan fitur yang paling memisahkan kelas target. Mudah dipahami dan diinterpretasikan, namun rawan overfitting jika tidak dikontrol.
<p align="center">
  <img src="https://github.com/user-attachments/assets/df697584-c2d7-4eee-a635-f08d04db006d" width="1000"/>
</p>

True Positives (TP): 3785
Kasus benar yang berhasil dikenali sebagai terkena penyakit kardiovaskular.

True Negatives (TN): 3601
Kasus benar yang dikenali sebagai tidak terkena.

False Positives (FP): 3169
Kasus yang sebenarnya sehat namun diklasifikasi terkena.

False Negatives (FN): 3225
Kasus yang sebenarnya terkena penyakit namun diklasifikasi sebagai tidak terkena — ini cukup berisiko.


### 6. Gradient Boosting
Model ensemble yang membangun pohon keputusan secara bertahap, memperbaiki kesalahan dari model sebelumnya. Gradient Boosting sangat kuat dan sering digunakan dalam kompetisi karena kemampuannya menangani kompleksitas data.
<p align="center">
  <img src="https://github.com/user-attachments/assets/01b23ef5-6f3a-4d94-80d0-bc4bbf279488" width="1000"/>
</p>
Insight :

True Positives (TP): 6147
Kasus yang benar dikenali sebagai terkena penyakit kardiovaskular.

True Negatives (TN): 3684
Kasus yang benar dikenali sebagai tidak terkena.

False Positives (FP): 3086
Kasus sehat yang diklasifikasikan sebagai terkena.

False Negatives (FN): 863
Kasus yang terkena namun tidak dikenali — jumlah ini jauh lebih sedikit dibanding model lain.

### 7. AdaBoost
Algoritma boosting lain yang menggabungkan beberapa weak learner (biasanya decision stump) secara iteratif. Fokus pada kesalahan sebelumnya dengan menyesuaikan bobot data yang salah klasifikasi.
<p align="center">
  <img src="https://github.com/user-attachments/assets/d48d048e-2394-433c-af6a-4ad0c3bf6155" width="1000"/>
</p>
Insight :

True Positives (TP): 6422 – kasus sakit yang berhasil dikenali.

False Negatives (FN): 588 – kasus sakit yang tidak terdeteksi (relatif rendah).

True Negatives (TN): 2896 – kasus sehat yang dikenali dengan benar.

False Positives (FP): 3874 – kasus sehat yang salah dikenali sebagai sakit.

### 8. Naive Bayes
Model probabilistik berdasarkan Teorema Bayes yang mengasumsikan independensi antar fitur. Cocok untuk klasifikasi cepat dan bekerja baik pada data yang relatif sederhana.
<p align="center">
  <img src="https://github.com/user-attachments/assets/2d7f1463-8637-4852-b380-026e61f7a310" width="1000"/>
</p>
Insight :

True Positives (TP): 5569 – kasus sakit terdeteksi benar.

False Negatives (FN): 1441 – kasus sakit tidak terdeteksi.

True Negatives (TN): 3929 – kasus sehat dikenali dengan benar.

False Positives (FP): 2841 – kasus sehat yang salah dideteksi sebagai sakit.



## Kesimpulan

### 1. Mengetahui seberapa akurat model machine learning dalam memprediksi risiko penyakit kardiovaskular berdasarkan variabel-variabel kesehatan seperti tekanan darah, kadar kolesterol, kadar glukosa, serta faktor gaya hidup.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f2d70b37-2f63-4288-b878-243642a29b7c" width="600"/>
</p>

**Hasil Prediksi:**

✅ Prediksi: ⚠️ Terkena Cardiovascular

✅ Probabilitas: 75%

**📌 Insight:**

- Meskipun tekanan darah dan glukosa terlihat normal, tinggi badan dan berat badan menyebabkan BMI yang sedikit di atas normal, serta kolesterol di atas normal (2) dapat menjadi faktor risiko yang signifikan.

- Model mengidentifikasi pasien ini sebagai berisiko tinggi terkena penyakit kardiovaskular, dengan keyakinan cukup tinggi (84%), mengindikasikan model sensitif terhadap fitur-fitur kesehatan utama seperti kolesterol dan BMI.


### 2. Mengetahui apakah jenis kelamin dan kelompok usia tertentu memiliki risiko lebih tinggi terhadap penyakit kardiovaskular

<p align="center">
  <img src="https://github.com/user-attachments/assets/ae2c2bdd-f66d-4218-81ce-c10e1c7c2b1a" width="600"/>
</p>

**📊 Interpretasi dari Grafik**

- Kedua jenis kelamin memiliki pola yang mirip: risiko meningkat seiring bertambahnya usia.

- Pria usia 60-69 memiliki proporsi tertinggi terkena penyakit kardiovaskular (~68%).

- Wanita usia 60-69 juga sangat tinggi (~64%), sedikit lebih rendah dari pria.

- Kelompok usia di bawah 40 memiliki proporsi risiko yang jauh lebih rendah (< 25%) untuk kedua gender.

**Kesimpulan**

- Risiko meningkat secara signifikan setelah usia 40 tahun, paling tinggi di kelompok 60-69 tahun.

- Pria memiliki sedikit risiko lebih tinggi dibanding wanita pada semua kelompok usia.
