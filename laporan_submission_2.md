# Laporan Proyek Machine Learning - Meicha Salsabila Budiyanti

## Project Overview

Pertumbuhan industri hiburan digital, khususnya dalam bidang drama Korea (K-Drama), telah mengalami perkembangan pesat dalam beberapa tahun terakhir. Popularitas K-Drama yang semakin mendunia menyebabkan meningkatnya jumlah rilisan setiap tahunnya. Kondisi ini menimbulkan information overload, yaitu situasi di mana pengguna menghadapi terlalu banyak pilihan sehingga kesulitan dalam menentukan tontonan yang sesuai dengan preferensi mereka (Ricci, Rokach, & Shapira, 2015).

Salah satu solusi efektif untuk mengatasi permasalahan ini adalah dengan menggunakan sistem rekomendasi. Sistem rekomendasi mampu memberikan saran personal berdasarkan histori interaksi pengguna terhadap konten, dan telah diimplementasikan secara luas pada platform seperti Netflix, YouTube, dan Spotify (Su & Khoshgoftaar, 2009). Dalam konteks K-Drama, sistem rekomendasi tidak hanya meningkatkan kepuasan pengguna, tetapi juga berdampak pada retensi dan peningkatan engagement terhadap platform streaming.

Pendekatan sistem rekomendasi konvensional seperti popularity-based atau content-based filtering memiliki keterbatasan karena kurang mampu menangkap preferensi kolektif pengguna (Schäfer et al., 2007). Sebaliknya, Collaborative Filtering (CF) memanfaatkan interaksi antar pengguna untuk mengidentifikasi pola dan memberikan rekomendasi yang lebih relevan.

Dalam proyek ini, dikembangkan sebuah sistem rekomendasi drama Korea berbasis Neural Collaborative Filtering (NCF) menggunakan TensorFlow dan Keras. Model ini memanfaatkan data interaksi pengguna dan drama berupa rating yang terdapat pada dataset reviews.csv. Dataset tambahan korean_drama.csv dan wiki_actors.csv digunakan untuk eksplorasi dan analisis, tetapi model difokuskan pada relasi user-item berdasarkan rating.

Model dirancang dengan menggunakan embedding layers untuk merepresentasikan masing-masing pengguna dan drama dalam bentuk vektor berdimensi rendah. Representasi tersebut kemudian digabungkan dan diproses melalui beberapa lapisan neural network untuk memodelkan hubungan non-linear antar pengguna dan item. Proses pelatihan dilakukan menggunakan pendekatan pembelajaran terawasi dengan target rating, dan model dievaluasi menggunakan metrik Root Mean Squared Error (RMSE) untuk mengukur akurasi prediksi.

Sistem ini diharapkan mampu memberikan rekomendasi yang lebih personal, akurat, dan kontekstual, serta menjadi fondasi bagi pengembangan sistem rekomendasi berbasis deep learning yang lebih lanjut dalam domain hiburan digital, khususnya K-Drama.

## Business Understanding

Industri hiburan digital, khususnya drama Korea (K-Drama), berkembang sangat pesat dalam beberapa tahun terakhir. Dengan semakin banyaknya platform streaming seperti Netflix, Viu, dan iQIYI, jumlah K-Drama yang dirilis tiap tahun pun ikut meningkat. Kini, penonton bisa memilih dari ratusan judul dengan berbagai genre dan aktor yang berbeda-beda.

Namun, banyaknya pilihan ini justru bisa membuat bingung. Pengguna sering merasa kesulitan menentukan tontonan mana yang paling cocok dengan selera mereka. Terlalu banyak pilihan membuat sebagian orang kewalahan dan akhirnya malah tidak tahu harus menonton apa. Masalah ini dikenal sebagai information overload, yaitu kondisi ketika terlalu banyak informasi membuat seseorang sulit mengambil keputusan. Oleh karena itu, dibutuhkan sistem yang bisa membantu pengguna dalam memilih tontonan yang sesuai dengan minat mereka secara lebih mudah dan cepat.

Dalam proyek ini, dibangun sebuah sistem rekomendasi anime berbasis pembelajaran mesin yang bertujuan untuk membantu pengguna menemukan anime yang relevan dengan minat dan histori penontonan mereka. Sistem rekomendasi ini berfokus pada dua pendekatan utama, yaitu Collaborative Filtering dan Content-Based Filtering.

### Problem Statements

1. Terlalu banyak pilihan K-Drama membuat pengguna kesulitan menemukan tontonan yang sesuai dengan preferensi mereka.

2. Bagaimana menyajikan rekomendasi tontonan K-Drama yang sesuai dengan preferensi unik setiap pengguna?
   
3. Bagaimana memanfaatkan baik informasi konten drama maupun interaksi pengguna sebelumnya untuk menghasilkan rekomendasi yang akurat?

4. Kurangnya pemanfaatan data histori pengguna secara efektif dalam memberikan rekomendasi yang personal.

### Goals

1. Mengembangkan sistem rekomendasi untuk K-Drama yang mampu memberikan saran tontonan secara personal dan relevan.
 
2. Memberikan pengalaman pengguna yang lebih baik dan meningkatkan keterlibatan pengguna terhadap platform penyedia K-Drama.
   
3. Mengimplementasikan dua pendekatan utama sistem rekomendasi, yaitu Collaborative Filtering dan Content-Based Filtering, untuk membandingkan performa keduanya dalam menghasilkan rekomendasi.
   
4. Melakukan pengujian sistem rekomendasi dengan mengevaluasi akurasi hasil rekomendasi berdasarkan data rating pengguna.

### Solution Statements

Untuk mencapai tujuan di atas, berikut adalah dua pendekatan solusi yang digunakan dalam proyek ini:
1. Collaborative Filtering :Pendekatan ini memberikan rekomendasi berdasarkan kemiripan fitur antar drama, seperti genre dan aktor. Sistem menghitung cosine similarity antara drama yang pernah ditonton pengguna dan drama lainnya, lalu merekomendasikan drama yang paling mirip. Data yang digunakan berasal dari file korean_drama.csv dan wiki_actors.csv. Fitur-fitur penting (genre dan aktor) diolah dan digunakan untuk membentuk representasi teks (TF-IDF). Cosine similarity digunakan untuk mengukur kemiripan antar drama.

2. Content-Based Filtering : Pendekatan ini memberikan rekomendasi dengan memanfaatkan interaksi pengguna sebelumnya, khususnya rating. NCF digunakan untuk menangkap pola non-linear dalam preferensi pengguna. Data interaksi pengguna diambil dari reviews.csv. Model dibangun menggunakan TensorFlow dan memanfaatkan embedding layers untuk mewakili pengguna dan item dalam ruang vektor berdimensi rendah. Prediksi skor rating dilakukan melalui dot product dari vektor embedding pengguna dan item, serta bias. Model dievaluasi menggunakan Root Mean Squared Error (RMSE).

## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan gabungan dari tiga sumber utama, yaitu:

* korean_drama.csv – berisi informasi deskriptif tentang drama Korea.

* reviews.csv – berisi ulasan pengguna terhadap drama.

* wiki_actors.csv – berisi informasi aktor yang membintangi drama.

Dataset ini diambil dari [Kaggle](https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist). Terdapat tiga file berformat csv pada dataset, yaitu korean_drama.csv, reviews.csv, dan wiki-actors.csv. Beberapa rincian penjelasan dataset ini sebagai berikut:

### Statistika Dataset

| Nama File          | Jumlah Baris | Jumlah Kolom | Deskripsi Singkat                                   |
| ------------------ | ------------ | ------------ | --------------------------------------------------- |
| korean_drama.csv | 1.752        | 17           | Metadata drama Korea 2015–2023                      |
| reviews.csv      | 10.625       | 10           | Ulasan dan rating pengguna                          |
| wiki_actors.csv  | 8.659        | 5            | Nama aktor dan karakter yang diperankan dalam drama |


### Deskripsi Fitur

1. Dataset Korean Drama (korean_drama.csv)
   
   | Fitur         | Tipe Data | Deskripsi                                                |
   | ------------- | --------- | -------------------------------------------------------- |
   | kdrama_id     | Object    | ID unik tiap drama                                       |
   | drama_name    | Object    | Judul drama                                              |
   | year          | Integer   | Tahun rilis                                              |
   | director      | Object    | Nama sutradara                                           |
   | screenwriter  | Object    | Penulis naskah                                           |
   | country       | Object    | Negara asal drama (biasanya Korea Selatan)              |
   | type          | Object    | Tipe tayangan (TV, Web, dll.)                            |
   | tot_eps       | Integer   | Jumlah episode                                           |
   | duration      | Float     | Durasi rata-rata per episode (menit)                     |
   | start_dt      | Object    | Tanggal mulai tayang                                     |
   | end_dt        | Object    | Tanggal selesai tayang                                   |
   | aired_on      | Object    | Hari tayang                                              |
   | org_net       | Object    | Jaringan TV/Platform penayang                            |
   | content_rt    | Object    | Rating konten (misalnya: 15+, 19+)                       |
   | synopsis      | Object    | Ringkasan cerita                                         |
   | rank          | Integer   | Peringkat popularitas                                    |
   | pop           | Integer   | Indeks popularitas (semacam skor gabungan MyDramaList)  |


2. Dataset Reviews (`reviews.csv`)
   
   | Fitur               | Tipe Data | Deskripsi                                            |
   | -------------------|-----------|------------------------------------------------------|
   | user_id            | Object    | ID pengguna                                          |
   | title              | Object    | Judul drama yang diulas                              |
   | story_score        | Float     | Skor cerita                                          |
   | acting_cast_score  | Float     | Skor akting dan pemain                               |
   | music_score        | Float     | Skor musik                                           |
   | rewatch_value_score| Float     | Skor niat menonton ulang                             |
   | overall_score      | Float     | Skor keseluruhan                                     |
   | review_text        | Object    | Isi ulasan pengguna                                  |
   | ep_watched         | Object    | Jumlah episode yang ditonton                         |
   | n_helpful          | Integer   | Jumlah pengguna yang merasa ulasan tersebut membantu |
   
3. Dataset Actors (wiki_actors.csv)

   | Fitur            | Tipe Data | Deskripsi                                 |
   | ---------------- | --------- | ----------------------------------------- |
   | actor_id       | Object   | ID unik aktor                             |
   | actor_name     | Object    | Nama aktor                                |
   | drama_name     | Object    | Judul drama yang dibintangi               |
   | character_name | Object    | Nama karakter yang diperankan             |
   | role           | Object    | Peran dalam drama (utama, pendukung, dll) |

1. Informasi Struktur Data
   - Fungsi drama.info() dan review.info() digunakan untuk memeriksa jumlah entri dan tipe data.
   - Dataset drama berisi informasi metadata drama Korea, sedangkan review berisi skor yang diberikan pengguna.
   - Dataset actors juga diperiksa strukturnya dengan actors.info().
   ```python
   drama.info()
   review.info()
   actors.info()
   ```
   
2. Nilai Unik dan Missing Value
   - Fungsi isnull().sum() digunakan pada ketiga dataset (drama, review, actors) untuk mengidentifikasi missing value.
   - Teridentifikasi kolom-kolom dengan missing seperti duration, rank, dan org_net pada dataset drama.
   - Nilai duplikat juga dicek dengan duplicated().sum().
   ```python
   drama.isnull().sum()
   review.isnull().sum()
   actors.isnull().sum()
   
   drama.duplicated().sum()
   review.duplicated().sum()
   actors.duplicated().sum()
   ```

3. Statistika Deskriptif
   - Statistik deskriptif dilakukan dengan describe(include='all') untuk semua dataset.
   - Diketahui bahwa jumlah episode, durasi, dan rating memiliki distribusi yang sangat bervariasi, mencerminkan keragaman konten drama Korea.
   ```python
   drama.describe(include='all')
   review.describe(include='all')
   actors.describe(include='all')
   ```

4. Jumlah Drama per Tahun
   - value_counts().sort_index() digunakan untuk menghitung jumlah drama per tahun dan divisualisasikan dalam bar chart.
   - Menunjukkan tren peningkatan produksi drama dari tahun ke tahun.
   ```python
   drama['year'].value_counts().sort_index().plot(kind='bar', figsize=(12,5))
   plt.title('Jumlah Drama per Tahun')
   plt.xlabel('Tahun')
   plt.ylabel('Jumlah Drama')
   plt.show()
   ```

5. Distribusi Rating oleh Pengguna
   - review['rating'] = review['overall_score'] digunakan untuk konsistensi nama kolom.
   - Distribusi skor pengguna divisualisasikan dengan bar chart, menunjukkan mayoritas pengguna memberikan rating tinggi.
   ```python
   review['rating'] = review['overall_score']
   review['rating'].value_counts().sort_index().plot(kind='bar', figsize=(10,4))
   plt.title('Distribusi Rating oleh Pengguna')
   plt.xlabel('Skor')
   plt.ylabel('Jumlah')
   plt.show()
   ```

6. Distribusi Rata-Rata Skor per Drama
   - Rata-rata skor dihitung dengan groupby('title')['overall_score'].mean() dan divisualisasikan.
   - Sebagian besar drama memiliki skor rata-rata antara 7 dan 9.
   ```python
   drama.describe(include='all')
   review.describe(include='all')
   actors.describe(include='all')
   ```

7. Distribusi Tipe Drama dan Jaringan
   - Distribusi jenis drama (type) dan negara produksi (country) dihitung dengan value_counts().
   - Jaringan penyiaran teratas divisualisasikan, dengan jaringan seperti tvN, JTBC, dan KBS mendominasi produksi.
   ```python
   review.groupby('title')['overall_score'].mean().plot(kind='hist', bins=20, figsize=(10,5))
   plt.title('Distribusi Rata-Rata Skor per Drama')
   plt.xlabel('Rata-Rata Skor')
   plt.ylabel('Jumlah Drama')
   plt.show()
   ```

8. Visualisasi Rating Konten
   - Distribusi rating konten (content_rt) divisualisasikan dengan bar chart.
   ```python
   drama['content_rt'].value_counts().plot(kind='bar', title='Distribusi Rating Konten')
   plt.ylabel('Jumlah Drama')
   plt.show()
   ```

9. Aktor dan Peran
    - Distribusi peran dalam dataset actors divisualisasikan.
    - 10 aktor dengan frekuensi kemunculan terbanyak ditampilkan menggunakan bar chart horizontal.
   ```python
    actors['role'].value_counts().plot(kind='bar', title='Distribusi Peran Aktor')
   plt.ylabel('Jumlah')
   plt.show()
   
   actors['actor_name'].value_counts().head(10).sort_values().plot(kind='barh', title='10 Aktor Terbanyak')
   plt.xlabel('Jumlah Drama')
   plt.show()
   ```

## Data Preparation

Pada tahap ini, dilakukan serangkaian proses untuk membersihkan dan menyiapkan data sebelum digunakan dalam pemodelan machine learning. Langkah-langkah preprocessing disusun secara sistematis agar data yang digunakan berkualitas dan konsisten.

1. Menyalin Dataset Asli
   Dataset utama disalin terlebih dahulu agar proses pembersihan tidak memengaruhi data mentah.
   ```python
   drama_clean = drama.copy()
   review_clean = review.copy()
   actors_clean = actors.copy()
   ```

2. Menghapus Nilai Kosong (_Missing Value_)
   ```python
   drama_clean['synopsis'] = drama_clean['synopsis'].fillna('')
   drama_clean['aired_on'] = drama_clean['aired_on'].fillna('Unknown')
   drama_clean['org_net'] = drama_clean['org_net'].fillna('Unknown')
   drama_clean['duration'] = drama_clean['duration'].fillna(drama_clean['duration'].median())
   review_clean['review_text'] = review_clean['review_text'].fillna('')
   ```
   Baris dengan nilai kosong dihapus untuk menghindari error atau hasil yang tidak akurat selama pemodelan. Fitur seperti genre, type, rating, dan members sangat penting untuk pemrosesan lebih lanjut (terutama pada Content-Based Filtering), sehingga missing value perlu dihilangkan.

3. Menghapus Kolom dengan Banyak Missing Value
   ```python
   drama_clean.drop(columns=['director', 'screenwriter'], inplace=True)
   ```
   Penghapusan ini dilakukan agar data tidak tercemar oleh fitur yang tidak lengkap dan dapat menyebabkan noise pada model.

4. Reset Index Setelah Pembersihan
   ```python
   drama_clean.reset_index(drop=True, inplace=True)
   review_clean.reset_index(drop=True, inplace=True)
   ```
   Index di-reset setelah data dibersihkan agar struktur data lebih rapi dan siap digabungkan di tahap selanjutnya.

5. Sinkronisasi Review dengan Data Drama
   ```python
   review_clean = review_clean[review_clean['title'].isin(drama_clean['drama_name'])]
   ```
   Review yang tidak memiliki kecocokan dengan judul drama dalam dataset utama dihapus untuk menjaga konsistensi antar data. Langkah ini memastikan bahwa hanya review yang benar-benar relevan dengan drama yang tersedia yang digunakan.

6. Sampling Review
   ```python
   review_per_drama = review_clean.groupby('title').apply(lambda x: x.sample(min(len(x), 100), random_state=42)).reset_index(drop=True)
   ```
   Untuk mencegah dominasi dari drama tertentu yang memiliki terlalu banyak review, dilakukan sampling maksimal 100 review per drama. Sampling ini menjaga distribusi data tetap seimbang dan mempercepat proses pelatihan model.

7. Penggabungan Dataset

   Beberapa penggabungan dilakukan untuk menyatukan data dari berbagai sumber agar analisis dan pemodelan dapat dilakukan dengan menyeluruh.
   
   ```python
   # Gabungkan review dengan info drama
   drama_review = pd.merge(review_per_drama, drama_clean, left_on='title', right_on='drama_name')
   
   # Gabungkan dengan data aktor
   full_data = pd.merge(drama_review, actors, on='drama_name', how='left')

   Penggabungan ini penting untuk membuat satu dataset terpadu yang mengandung informasi drama, ulasan, dan aktor dalam satu struktur data yang siap digunakan dalam modeling. Dataset ini berisi informasi lengkap yang dibutuhkan untuk model Collaborative Filtering (berbasis user-anime-rating) dan Content-Based Filtering (berbasis fitur seperti genre dan type).

8. Menghapus Nilai Kosong pada Data Gabungan
   ```python
   full_data.dropna(inplace=True)
   ```
   Setelah semua data digabungkan, sisa nilai kosong yang ada dihapus agar dataset benar-benar bersih.. Langkah ini dilakukan untuk memastikan tidak ada missing value yang tersisa dan struktur data sudah siap digunakan pada tahap modeling.

9. Final Check
   ```python
   full_data.info()
   full_data.isnull().sum()
   full_data.head()
   ```
   Pemeriksaan akhir dilakukan untuk memastikan bahwa struktur data sudah bersih dan siap digunakan. Pemeriksaan ini memastikan tidak ada nilai kosong yang tersisa dan data sudah dalam format yang sesuai.
   
### Content Based Filtering
1. Menghapus Duplikat dan Mengisi Kolom Kosong
   ```python
   drama_content = drama_content.drop_duplicates(subset='drama_name', keep='first').reset_index(drop=True)
   drama_content['combined'] = drama_content['review_text'].fillna('') + ' ' + \
                                drama_content['actor_name'].fillna('') + ' ' + \
                                drama_content['title'].fillna('')
   ```
   Dataset drama_content merupakan gabungan ulasan, aktor, dan judul dari setiap drama. Duplikat berdasarkan drama_name dihapus untuk memastikan setiap drama hanya muncul sekali. Selanjutnya, kolom teks yang masih kosong diisi dengan string kosong ('') agar bisa diproses dalam tahap tokenisasi.
   
2. TF-IDF
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   tfidf = TfidfVectorizer(stop_words='english')
   tfidf_matrix = tfidf.fit_transform(drama_content['combined'])
   ```
   Metode TF-IDF digunakan untuk mengubah kolom gabungan teks (review_text, actor_name, title) menjadi representasi numerik. TF-IDF membantu mengidentifikasi kata-kata yang paling penting dan khas dari tiap drama, sambil mengabaikan kata-kata umum (stop words). Representasi ini diperlukan untuk menghitung kemiripan antar drama.

3. Cosine Similarity
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
   ```
   Cosine Similarity digunakan untuk mengukur sejauh mana dua drama memiliki konten yang mirip berdasarkan vektor TF-IDF. Hasil dari cosine similarity ini menjadi dasar sistem rekomendasi drama yang mirip.

4. Index Mapping
   ```python
   indices = pd.Series(drama_content.index, index=drama_content['drama_name'].str.lower())
   ```
   Langkah ini memetakan judul drama ke indeks baris yang sesuai dalam dataset. Hal ini penting agar sistem rekomendasi dapat dengan cepat mencari dan memberikan rekomendasi berdasarkan kemiripan dengan drama lainnya. Mapping ini memungkinkan pencarian yang efisien dan akurat.

### Colaborative Filtering
1. Encoding
    ```python
   user_ids = df['user_id'].unique().tolist()
   title_ids = df['title'].unique().tolist()
   
   user_to_index = {x: i for i, x in enumerate(user_ids)}
   title_to_index = {x: i for i, x in enumerate(title_ids)}
   
   df['user'] = df['user_id'].map(user_to_index)
   df['title_id'] = df['title'].map(title_to_index)
    ```
    Kode ini mengubah ID pengguna (user_id) dan ID judul drama (title) menjadi indeks numerik yang berurutan mulai dari 0. Proses ini diperlukan dalam collaborative filtering karena algoritma ini membutuhkan input numerik untuk membangun matriks interaksi pengguna-drama. Dengan menggunakan indeks numerik, data lebih mudah dikelola dalam model rekomendasi.

2. Splitting Data
    ```python
   X = df[['user', 'title_id']]
   y = df['overall_score']
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
    Data kemudian dibagi menjadi data latih (training set) dan data uji (testing set) dengan rasio 80:20 menggunakan train_test_split. Pembagian ini penting agar model dapat dilatih menggunakan sebagian data dan dievaluasi menggunakan data yang belum pernah dilihat sebelumnya, untuk menghindari overfitting.

## Modeling

Dalam proyek ini, saya membangun dua pendekatan berbeda untuk sistem rekomendasi drama, yaitu **Content-Based Filtering** dan **Collaborative Filtering**. Masing-masing pendekatan dirancang untuk menyajikan rekomendasi yang personal dan relevan kepada pengguna berdasarkan data yang tersedia.

---

### 1. Content-Based Filtering

Content-Based Filtering merekomendasikan drama yang memiliki kemiripan berdasarkan konten—dalam hal ini, teks gabungan dari **review**, **nama aktor**, dan **judul** drama. Sistem ini hanya memerlukan informasi konten dari item, tanpa perlu mengetahui preferensi pengguna lain.

**Langkah-langkah:**
- Menggabungkan fitur teks (`review_text`, `actor_name`, `title`) ke dalam satu kolom (`combined`).
- Mengubah teks gabungan menjadi representasi numerik menggunakan **TF-IDF Vectorizer**.
- Menghitung kemiripan antar drama menggunakan **Cosine Similarity**.
- Membangun fungsi `recommend()` untuk mengembalikan drama yang paling mirip berdasarkan input judul atau nama aktor.

**Kelebihan:**
- Tidak membutuhkan data pengguna (cocok untuk cold start pada item baru).
- Dapat menjelaskan alasan rekomendasi (misalnya karena genre atau kontennya mirip).

**Kekurangan:**
- Hanya merekomendasikan drama yang mirip kontennya—tidak menangkap preferensi kompleks pengguna.
- Tidak bisa menyarankan drama dari genre yang belum pernah ditonton pengguna.

**Cosine Similarity Formula:**
$$\text{cosine similarity} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \cdot \|\vec{B}\|}$$

**Contoh Penerapan:**
```python
recommend('Crash Landing on You', 3)
```

### 2. Collaborative Filtering

Collaborative Filtering mempelajari hubungan antara pengguna dan drama berdasarkan rating yang diberikan, tanpa melihat isi konten drama itu sendiri. Model ini dibangun menggunakan pendekatan **Matrix Factorization** berbasis neural network dengan TensorFlow/Keras.

**Langkah-langkah:**
- Melakukan encoding `user_id` dan `title` menjadi ID numerik (`user`, `title_id`).
- Membagi data menjadi data latih dan uji (80:20).
- Membangun model dengan embedding layer untuk pengguna dan drama.
- Melatih model untuk mempelajari representasi laten dari pengguna dan drama.

**Kelebihan:**
- Dapat menangkap preferensi pengguna secara personal.
- Tidak terbatas pada kemiripan konten—bisa memberikan rekomendasi yang lebih beragam.
- Efektif dalam memahami pola perilaku pengguna dari data rating.

**Kekurangan:**
- Kurang efektif jika pengguna baru (cold start).
- Memerlukan data interaksi yang cukup dan proses training lebih kompleks.

#### Langkah-langkah:
- Melakukan encoding `user_id` dan `title` menjadi ID numerik (`user`, `title_id`).
- Membagi data menjadi data latih dan uji (80:20).
- Membangun model dengan embedding layer untuk pengguna dan drama.
- Melatih model untuk mempelajari representasi laten dari pengguna dan drama.

#### Arsitektur Model:
- Embedding Layer untuk pengguna dan drama (`Embedding(input_dim, output_dim)`).
- Kombinasi vektor embedding menggunakan dot product.
- Dense layer untuk pemrosesan lebih lanjut.
- Output layer dengan aktivasi linear.
  ```python
  user_embedding = layers.Embedding(input_dim=n_users, output_dim=embedding_size)(user_input)
  title_embedding = layers.Embedding(input_dim=n_titles, output_dim=embedding_size)(title_input)

  dot_product = layers.Dot(axes=1)([layers.Flatten()(user_embedding), layers.Flatten()(title_embedding)])
  output = layers.Dense(1)(dot_product)
  ```

   * Cara Kerja
      * Input Data: Pasangan (user, title) diubah menjadi indeks numerik.
      * Embedding Layer: Masing-masing ID diubah menjadi vektor berdimensi tetap (50).
      * Dot Product: Embedding dari user dan title dikombinasikan menggunakan dot product.
      * Output Layer: Dense layer dengan 1 neuron tanpa fungsi aktivasi, karena prediksi berada pada skala rating asli.
        
   * Pelatihan Model:
Model dilatih menggunakan loss function Mean Squared Error (MSE) dan optimizer Adam. Evaluasi dilakukan dengan metrik Root Mean Squared Error (RMSE). Penggunaan EarlyStopping dan ReduceLROnPlateau membantu menghentikan pelatihan saat performa stagnan dan menyesuaikan learning rate.
   * Contoh Training:
     ```python
     history = model.fit([X_train['user'], X_train['title_id']], y_train,validation_data=([X_test['user'], X_test['title_id']],
     y_test),epochs=50, batch_size=32)
     ```
  * Contoh Rekomendasi:

    Drama rated highly by user:
    - Sing My Crush | Rating: 9.0
    - Our Dating Sim | Rating: 9.0
    - Roommates of Poongduck 304 | Rating: 9.0
    - Semantic Error | Rating: 9.0
    - Twenty-Five Twenty-One | Rating: 9.0
    - Where Your Eyes Linger | Rating: 9.0
    - Blueming | Rating: 8.5
    - Unlock My Boss | Rating: 8.5
    - Ghost Doctor | Rating: 8.5
    - Big Mouth | Rating: 8.5
    - Wish You: Your Melody From My Heart | Rating: 8.0
    - The Golden Spoon | Rating: 8.0
    - The School Nurse Files | Rating: 8.0
    - Itaewon Class | Rating: 8.0
    - Happiness | Rating: 8.0
    - Bad and Crazy | Rating: 8.0
    - Dali and the Cocky Prince | Rating: 8.0
    - My Sweet Dear | Rating: 7.0
    - The Director Who Buys Me Dinner | Rating: 7.0
    - Happy Merry Ending | Rating: 6.5
    - All of Us Are Dead | Rating: 6.5
    - Grid | Rating: 6.5
    - Lovers of the Red Sky | Rating: 6.5
    - Duty After School: Part 2 | Rating: 4.0
    
    ==========================================================================
    
    Top 10 Drama Recommendations:
    - The King of Pigs
    - XX
    - Missing: The Other Side
    - She Makes My Heart Flutter
    - The Eighth Sense
    - We're Not Trash
    - Ending Again
    - The Veil
    - Stranger Season 2
    - Insider
   
## Evaluation

### Content-Based Filtering

Evaluasi pada pendekatan Content-Based Filtering dilakukan secara kualitatif dengan cara menguji apakah sistem mampu merekomendasikan drama yang relevan terhadap input pengguna. Evaluasi ini dilakukan dengan mengeksekusi fungsi `recommend()` pada beberapa drama populer dan memeriksa hasilnya secara manual.

**Contoh Pengujian:**
```python
recommend("Crash Landing on You", 3)
```
Hasil rekomendasi menunjukkan bahwa sistem berhasil memberikan drama yang secara konten mirip dengan input, seperti genre romance, drama, dan latar militer atau supernatural. Ini menandakan bahwa TF-IDF Vectorizer yang digunakan mampu menangkap informasi penting dari fitur teks.

### Collaborative Filtering

Dalam proyek ini digunakan **Root Mean Squared Error (RMSE)** sebagai metrik evaluasi utama. RMSE digunakan karena model berfokus pada prediksi nilai rating pengguna terhadap drama, yang merupakan data numerik kontinu.

#### Rumus RMSE:
$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

- $y_i$: rating aktual dari pengguna
- $\hat{y}_i$: rating hasil prediksi model
- $n$: jumlah sampel data

#### Cara Kerja RMSE:
- Mengukur jarak rata-rata kuadrat dari prediksi terhadap nilai asli.
- Kesalahan dikalikan dengan dirinya sendiri agar positif.
- Nilai akhir diakarkan agar satuannya kembali setara dengan rating asli.
- Karena rating telah dinormalisasi ke 0–1, maka nilai RMSE juga akan berada pada rentang ini.

#### Alasan Penggunaan RMSE:
- Sangat cocok untuk tugas regresi seperti prediksi rating.
- Memberikan penalti lebih besar untuk prediksi yang sangat meleset.
- Memudahkan interpretasi karena hasilnya masih berada dalam skala rating.

#### Visualisasi Hasil:
Notebook memplot perkembangan RMSE pada data training dan validation selama proses pelatihan: 

```python
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model RMSE over Epochs')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

### Analisis Terhadap Bussines Understanding
1. Apakah Model Menjawab Setiap Problem Statement?
   - Dalam konteks industri hiburan digital, khususnya K-Drama, pengguna sering mengalami information overload akibat terlalu banyaknya pilihan tontonan di berbagai platform streaming.
   - Pendekatan Content-Based Filtering membantu dengan menyarankan drama yang mirip berdasarkan konten seperti genre dan aktor, sehingga pengguna dapat menemukan tontonan baru yang sesuai dengan selera mereka.
   - Sementara itu, Collaborative Filtering mempelajari pola perilaku pengguna lain dengan preferensi serupa untuk memberikan rekomendasi yang lebih personal. Dengan kombinasi kedua pendekatan ini, sistem mampu menyederhanakan proses pencarian dan meningkatkan relevansi rekomendasi bagi pengguna.

2. Apakah Setiap Goals Berhasil Dicapai?
   - Model Collaborative Filtering telah dibangun dan dilatih menggunakan histori rating pengguna, menunjukkan performa yang baik melalui evaluasi menggunakan Root Mean Squared Error (RMSE).
   - Model Content-Based Filtering juga telah berhasil diimplementasikan dan dievaluasi secara kualitatif. Rekomendasi yang diberikan sesuai dengan konten yang diminati pengguna.
   - Sistem ini mampu menyaring ratusan pilihan menjadi beberapa tontonan yang relevan, yang menjadi salah satu tujuan utama proyek.

3. Apakah Solusi yang Direncanakan Memberikan Dampak?
   - Sistem rekomendasi yang dibangun berperan dalam membantu pengguna memilih tontonan dengan lebih cepat dan sesuai minat, yang mengurangi beban mental akibat informasi berlebih.
   - Rekomendasi berbasis histori dan preferensi membuat pengalaman pengguna menjadi lebih personal dan menyenangkan.
   - Proyek ini juga membuka peluang pengembangan lebih lanjut, seperti mengintegrasikan sinopsis, ulasan visual, atau perilaku tontonan real-time untuk meningkatkan kualitas rekomendasi.

### Kesimpulan
Secara keseluruhan, model yang dibangun telah berhasil menjawab seluruh problem statements, mencapai goals yang ditetapkan, dan solusi yang dirancang terbukti memberikan dampak yang nyata, meskipun dengan tingkat akurasi yang bervariasi. Collaborative Filtering menunjukkan performa yang lebih unggul dan efektif dalam memberikan rekomendasi yang personal, sedangkan Content-Based Filtering memberikan pendekatan alternatif yang masih memiliki ruang untuk ditingkatkan. Dengan demikian, proyek ini telah menunjukkan kontribusi nyata dalam meningkatkan pengalaman pengguna dalam menemukan anime yang sesuai dengan preferensi mereka.

## Referensi
> 
> Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook (2nd ed.). Springer. https://doi.org/10.1007/978-1-4899-7637-6
> 
> Schäfer, J. B., Frankowski, D., Herlocker, J. L., & Sen, S. (2007). Collaborative Filtering Recommender Systems. In The Adaptive Web (pp. 291–324). Springer. https://doi.org/10.1007/978-3-540-72079-9_9
> 
> Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. Advances in Artificial Intelligence, 2009, Article ID 421425. https://doi.org/10.1155/2009/421425
