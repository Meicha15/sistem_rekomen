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

### Exploratory Data Analysis (EDA)

1. Informasi Struktur Data
   * ```anime.info()``` dan ```rating.info()``` digunakan untuk mengecek jumlah entri (baris), tipe data setiap kolom, dan apakah ada nilai yang hilang (null).
   * Terdapat 12.294 anime dan 7.813.737 penilaian dari pengguna terhadap anime.
   * Data anime memiliki kolom bertipe objek dan numerik, sedangkan rating terdiri dari user_id, anime_id, dan rating bertipe integer.
   
2. Nilai Unik dan Missing Value
   * ```anime.nunique()``` digunakan untuk mengetahui jumlah nilai unik di setiap kolom.
   * Ditemukan missing value pada kolom genre, type, episodes, dan rating pada data anime.
   * Data rating tidak memiliki nilai yang hilang secara eksplisit (NaN), tetapi memiliki nilai -1 yang menandakan pengguna menonton anime namun tidak memberikan rating (ini umumnya dihapus dalam tahap preprocessing).
   
3. Eksplorasi Genre dan Type Anime
   * Genre dalam dataset ditulis dalam bentuk string yang dipisahkan koma.
   * Jumlah genre berbeda-beda dan kombinasi genre sangat beragam. Ini akan berguna untuk pendekatan Content-Based Filtering.
   * Jenis type anime meliputi TV, Movie, OVA, Special, ONA, dan beberapa lainnya.

4. Statistika Deskriptif
   Menggunakan ```anime.describe(include='all')``` dan ```rating.describe(include='all')``` untuk mengetahui:
   * Rata-rata, standar deviasi, nilai minimum dan maksimum dari anime dan rating.
   * Jumlah anime yang ditambahkan oleh pengguna (members) sangat bervariasi, menunjukkan ketimpangan popularitas anime.

5. Visualisasi Distribusi Rating oleh Pengguna
   * Visualisasi barplot dari rating['rating'].value_counts() menunjukkan bahwa pengguna cenderung memberi rating tinggi (8–10).
   * Ini mencerminkan adanya positivity bias di kalangan pengguna (umum dalam sistem rating online).
   * Terdapat rating -1 yang seharusnya rating bernilai > 1.
     ![rating user](https://github.com/user-attachments/assets/f4ee9c22-c325-47de-8c1b-c19508afd3fd)

6. Visualisasi Distribusi Rating per Anime
   * Distribusi rating rata-rata per anime divisualisasikan menggunakan histogram dan KDE plot (```sns.histplot```).
   * Sebagian besar anime memiliki rating rata-rata antara 6 dan 8, yang berarti tidak banyak anime yang dianggap sangat buruk atau sangat bagus oleh komunitas secara umum.
     ![anime rat](https://github.com/user-attachments/assets/79a7323e-15aa-43c0-8834-c5ecc2c3fee8)

7. Genre Terbanyak
   * Menggunakan ```Counter``` dan ```pd.Series```, genre diekstrak dan dihitung kemunculannya.
   * Genre paling diminati adalah Comedy, Action, Adventure, Fantasy, dan lain-lain.
   * Genre ini nantinya menjadi fitur penting dalam sistem rekomendasi berbasis konten.
     ![genre](https://github.com/user-attachments/assets/3f10c6e0-9f82-46ee-a181-6141c8a0c9de)

8. Distribusi Tipe Anime
   * Visualisasi distribusi jenis type menunjukkan bahwa TV Series mendominasi dataset, disusul oleh Movie, OVA, dan lainnya.
   * Ini memberi wawasan penting: sistem rekomendasi dapat menyesuaikan preferensi pengguna terhadap tipe anime.
     ![type](https://github.com/user-attachments/assets/08600727-fa24-4a4b-8018-05244793e08d)

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

### Content Based Filtering
Dalam content based filtering digunakan metrik evaluasi Precision@K adalah metrik evaluasi dalam sistem rekomendasi yang mengukur seberapa relevan item yang direkomendasikan oleh model dalam daftar K teratas.

* Rumus untuk Precision@K

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan dalam K rekomendasi}}{\text{K}}
$$

* Cara Kerja: <br>
  * Diambil sampel 100 pengguna secara acak dari data.
  * Untuk setiap pengguna, dipilih satu anime yang pernah mereka beri rating tinggi (disukai) sebagai query anime.
  * Dari query tersebut, sistem menghasilkan 5 anime teratas yang direkomendasikan menggunakan metode content-based filtering.
  * Precision dihitung dengan membandingkan daftar 5 rekomendasi terhadap daftar anime yang benar-benar disukai oleh pengguna tersebut (ground truth).
  * Proses ini diulang untuk seluruh 100 pengguna, dan nilai Precision@5 dirata-rata untuk mendapatkan skor keseluruhan (Average Precision@5).

* Hasil Evaluasi
  Berdasarkan evaluasi terhadap 100 pengguna, diperoleh hasil:

$$
\text{Average Precision@K} = 0.0560
$$
  
  Artinya, secara rata-rata hanya 5,6% dari 5 rekomendasi teratas yang benar-benar sesuai dengan preferensi pengguna. Ini menunjukkan bahwa sistem mulai mampu menangkap preferensi pengguna, namun akurasinya masih perlu ditingkatkan.

### Collaborative Filtering
Dalam proyek ini digunakan Root Mean Squared Error (RMSE) sebagai metrik evaluasi utama. RMSE digunakan karena model ini berfokus pada prediksi nilai rating user terhadap anime, yang merupakan data kontinu (bukan klasifikasi).

* Rumus RMSE:

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
$$

- $y_i$ = rating sebenarnya
- $\hat{y}_i$ = rating yang diprediksi oleh model
- $(n)$ = jumlah data

* Hasil RMSE pada data training dan validasi:

  ![download](https://github.com/user-attachments/assets/f877aef8-6d68-4428-8465-b973d26033f6)

* Cara Kerja RMSE
  RMSE menghitung selisih antara rating prediksi dan aktual, kemudian mengkuadratkan selisih tersebut agar tidak saling meniadakan (positif-negatif), menjumlahkannya, lalu diambil akar rata-rata kuadratnya. Dengan demikian:
  * Semakin kecil nilai RMSE, semakin kecil rata-rata kesalahan prediksi model.
  * Karena nilai rating dinormalisasi dalam rentang 0–1, maka RMSE juga berada di rentang 0–1.

* Alasan Memilih RMSE
  * Cocok untuk data regresi seperti prediksi rating.
  * Peka terhadap error besar, sehingga model didorong untuk meminimalkan prediksi yang jauh meleset.
  * Memudahkan interpretasi karena memiliki satuan yang sama dengan skala rating.

### Analisis Terhadap Bussines Understanding
1. Apakah Model Menjawab Setiap Problem Statement?
   * Melalui sistem rekomendasi berbasis Content-Based dan Collaborative Filtering, pengguna kini diberikan rekomendasi yang lebih terfokus, sehingga dapat menyaring pilihan dari ribuan anime menjadi beberapa opsi yang paling relevan. Ini mengurangi beban pencarian secara manual.
   * Model Collaborative Filtering berhasil mempelajari pola rating pengguna dan memberikan rekomendasi berdasarkan preferensi pengguna lain yang serupa. Hal ini menghasilkan rekomendasi yang lebih personal dibandingkan sekadar mengandalkan daftar anime populer.
   * Melalui teknik embedding userID dan animeID, model Collaborative Filtering berhasil memanfaatkan histori rating pengguna dalam prediksi. Evaluasi menggunakan RMSE menunjukkan bahwa model mampu memprediksi rating dengan tingkat kesalahan yang relatif rendah, menandakan adanya pemanfaatan data histori yang efektif.

2. Apakah Setiap Goals Berhasil Dicapai?
   * Model Collaborative Filtering telah dibangun dan diuji menggunakan data histori rating, serta menunjukkan performa yang baik melalui metrik RMSE.
   * Kedua pendekatan telah berhasil diimplementasikan dan dievaluasi. Content-Based Filtering dievaluasi menggunakan Precision@5, sementara Collaborative Filtering menggunakan RMSE. Hasil evaluasi menunjukkan bahwa keduanya memberikan hasil yang berbeda, yang bisa digunakan untuk eksplorasi hibridisasi model di masa depan.
   * Evaluasi telah dilakukan secara kuantitatif:
     * Precision@5 = 0.056 untuk Content-Based Filtering.
     * RMSE pada Collaborative Filtering menunjukkan nilai kesalahan prediksi yang cukup rendah.

3. Apakah Solusi yang Direncanakan Memberikan Dampak?
   * Model ini terbukti mampu memberikan rekomendasi berbasis pola rating antar pengguna. RMSE yang rendah menunjukkan bahwa model dapat memprediksi rating dengan cukup baik, sehingga membantu pengguna menemukan anime baru yang sesuai dengan preferensinya. Ini sangat relevan dengan kebutuhan sistem rekomendasi yang personal.
   * Meskipun nilai Precision@5 masih rendah (5,6%), sistem ini menunjukkan kemampuan awal dalam menangkap kemiripan konten, terutama berdasarkan genre. Ini bisa menjadi pondasi awal untuk pengembangan sistem rekomendasi berbasis konten yang lebih kaya, seperti menggunakan sinopsis, studio, atau bahkan data audio-visual di masa depan.

### Kesimpulan
Secara keseluruhan, model yang dibangun telah berhasil menjawab seluruh problem statements, mencapai goals yang ditetapkan, dan solusi yang dirancang terbukti memberikan dampak yang nyata, meskipun dengan tingkat akurasi yang bervariasi. Collaborative Filtering menunjukkan performa yang lebih unggul dan efektif dalam memberikan rekomendasi yang personal, sedangkan Content-Based Filtering memberikan pendekatan alternatif yang masih memiliki ruang untuk ditingkatkan. Dengan demikian, proyek ini telah menunjukkan kontribusi nyata dalam meningkatkan pengalaman pengguna dalam menemukan anime yang sesuai dengan preferensi mereka.

## Referensi
> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. Proceedings of the 26th International Conference on World Wide Web, 173–182. https://doi.org/10.1145/3038912.3052569
> 
> Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook (2nd ed.). Springer. https://doi.org/10.1007/978-1-4899-7637-6
> 
> Schäfer, J. B., Frankowski, D., Herlocker, J. L., & Sen, S. (2007). Collaborative Filtering Recommender Systems. In The Adaptive Web (pp. 291–324). Springer. https://doi.org/10.1007/978-3-540-72079-9_9
> 
> Su, X., & Khoshgoftaar, T. M. (2009). A survey of collaborative filtering techniques. Advances in Artificial Intelligence, 2009, Article ID 421425. https://doi.org/10.1155/2009/421425
