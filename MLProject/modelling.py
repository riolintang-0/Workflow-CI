import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_kamus(file_path):
    """
    Membaca file CSV kamus yang memiliki header ('word', 'weight')
    dan mengembalikannya sebagai set untuk pencarian cepat.
    """
    try:
        # Baca CSV dengan header, lalu ambil hanya kolom 'word'
        df = pd.read_csv(file_path)
        # Pastikan kolom 'word' ada di file
        if 'word' not in df.columns:
            print(f"Error: File {file_path} tidak memiliki kolom 'word'.")
            return set()
        return set(df['word'].astype(str).tolist())
    except FileNotFoundError:
        print(f"Error: File kamus tidak ditemukan di {file_path}")
        return set()

# 1. Mengaktifkan MLflow Autologging
mlflow.sklearn.autolog()

# 2. Memuat Kamus Sentimen dari File
print("Memuat kamus sentimen...")
KAMUS_POSITIF = load_kamus('kamus/positive.csv')
KAMUS_NEGATIF = load_kamus('kamus/negative.csv')

# Validasi apakah kamus berhasil dimuat
if not KAMUS_POSITIF or not KAMUS_NEGATIF:
    print("Salah satu atau kedua file kamus gagal dimuat. Proses dihentikan.")
    exit()

print(f"Berhasil memuat {len(KAMUS_POSITIF)} kata positif dan {len(KAMUS_NEGATIF)} kata negatif.")

# 3. Memuat Data yang Sudah Dibersihkan
df = pd.read_csv('data/dataset_rs_processed.csv')
df.dropna(subset=['ulasan'], inplace=True)

# 4. Membuat Label Sentimen
def label_sentiment(ulasan):
    """Memberi skor dan label pada ulasan berdasarkan kamus yang dimuat."""
    ulasan_split = str(ulasan).split()
    skor_positif = sum(1 for kata in ulasan_split if kata in KAMUS_POSITIF)
    skor_negatif = sum(1 for kata in ulasan_split if kata in KAMUS_NEGATIF)

    if skor_positif > skor_negatif:
        return 1  # Positif
    elif skor_negatif > skor_positif:
        return 0  # Negatif
    else:
        return -1 # Netral

df['sentiment'] = df['ulasan'].apply(label_sentiment)
df_final = df[df['sentiment'] != -1].copy()

print(f"Data setelah pelabelan: {len(df_final)} ulasan (mengabaikan ulasan netral).")
print("Distribusi sentimen:")
print(df_final['sentiment'].value_counts())

# 5. Feature Engineering & Pelatihan Model
X = df_final['ulasan']
y = df_final['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

with mlflow.start_run():
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Naive Bayes dilatih dengan Akurasi: {accuracy:.4f}")

print("\nEksperimen selesai. Jalankan 'mlflow ui' untuk melihat hasilnya.")