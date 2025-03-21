import streamlit as st
import pandas as pd
import numpy as np
import hydralit_components as hc
import datetime
from streamlit_option_menu import option_menu
from google_play_scraper import Sort, reviews
import re
import string
import nltk
import os
import io
import time
import csv
import pickle
import joblib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend to 'Agg'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.exceptions import NotFittedError
from PIL import Image, ImageDraw, ImageOps
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report

# NAVBAR
#['Home','Dataset', 'Preprocessing', 'Predict']
st.set_page_config(layout='wide',initial_sidebar_state='collapsed',)

# Inisialisasi session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Spesifikasi menu utama
menu_data = [
    # {'icon': "far fa-copy", 'label': "Scrapping"},
    {'id': 'Data', 'icon': "🐙", 'label': "Dataset"},
    {'icon': "fas fa-tachometer-alt", 'label': "Preprocessing", 'ttip': "Ini adalah tooltip Dashboard!"},
    {'icon': "far fa-chart-bar", 'label': "Predict"},
]

over_theme = {'txc_inactive': '#FFFFFF'}

menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    #login_name='Logout',
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

# Atur session state.page berdasarkan menu yang dipilih
if menu_id == 'Home':
    st.session_state.page = 'Home'
elif menu_id == 'Scrapping':
    st.session_state.page = 'Scrapping'
elif menu_id == 'Data':
    st.session_state.page = 'Dataset'
elif menu_id == 'Preprocessing':
    st.session_state.page = 'Preprocessing'
elif menu_id == 'Predict':
    st.session_state.page = 'Predict'

# Mendapatkan id dari item menu yang diklik
# st.info(f"Menu Terpilih: {menu_id}")

def casefolding(text):
    text = text.lower()
    text = text.strip()
    return text

def cleaning(text):
    # Menghapus username (@username)
    text = re.sub(r'@\w+', '', text)
    # Menghapus hashtag (#hashtag)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'#\S+', '', text)
    # Menghapus URL (http:// atau www.)
    text = re.sub(r'http[s]?://\S+', '', text)  # Menghapus http atau https URL
    text = re.sub(r'www\.\S+', '', text)        # Menghapus www URL
    # Menghapus karakter yang bukan alfanumerik dan spasi
    text = re.sub(r'[^0-9A-Za-z ]', '', text)
    text = re.sub(r'[^0-9A-Za-z\s]', ' ', text)
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    # Mengganti dua atau lebih spasi dengan satu spasi
    text = re.sub(r'\s+', ' ', text)
    # Menghapus spasi di awal dan akhir teks
    text = text.strip()
    return text

col_names_abbeviation = ['before','after']
indo_abbreviation = pd.read_csv('kamus_singkatan.csv', delimiter=';', names=col_names_abbeviation)
indo_abbreviation.head()

def replace_abbreviations(text, abbreviation):
    # Buat dictionary dari dataframe indo_abbreviation
    abbreviation_dict = dict(zip(abbreviation['before'], abbreviation['after']))

    # Fungsi untuk mengganti kata dalam text
    def replace_words(text):
        words = text.split()
        new_words = [abbreviation_dict[word] if word in abbreviation_dict else word for word in words]
        return ' '.join(new_words)

    # Terapkan fungsi replace_words ke text
    text = replace_words(text)

    return text

col_names_slank_words = ['before','after']
indo_slank_words = pd.read_csv('kamusalay.csv', delimiter=',', names=col_names_slank_words)
indo_slank_words.head()

def replace_slank_words(text, slank_words):
    # Buat dictionary dari dataframe indo_abbreviation
    abbreviation_dict = dict(zip(slank_words['before'], slank_words['after']))

    # Fungsi untuk mengganti kata dalam text
    def replace_words(text):
        words = text.split()
        new_words = [abbreviation_dict[word] if word in abbreviation_dict else word for word in words]
        return ' '.join(new_words)

    # Terapkan fungsi replace_words ke setiap text
    text = replace_words(text)
    return text

stopwords_ind = stopwords.words('indonesian')
df_stopwords = pd.read_csv('short_word.csv')
more_stopwords = df_stopwords['short_words'].tolist()
stopwords_ind = stopwords_ind + more_stopwords
df_stopwords_combined = pd.DataFrame({'stopwords': stopwords_ind})
df_stopwords_combined.to_csv('list_stopwords.csv', index=False)
preserved_words = ["tidak", "jangan"]
def remove_stop_words(text, preserved_words=None):
    clean_words = []
    text = text.split()
    for word in text:
        # Periksa apakah kata ada dalam daftar kata yang ingin dipertahankan
        if preserved_words and word in preserved_words:
            clean_words.append(word)
        elif word not in stopwords_ind:
            clean_words.append(word)
    return " ".join(clean_words)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
custom_exceptions = {'lgbt'}

def stemming(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) if word not in custom_exceptions else word for word in words]
    result = ' '.join(stemmed_words)
    return result

def tokenize(text):
    text = word_tokenize(text)
    return text

def remove_whitespace_and_combine(text, tokens):
    if tokens and isinstance(tokens, list):  # Periksa apakah tokens bukan None dan merupakan list
        # Menghapus whitespace yang tidak perlu
        tokens = [token.strip() for token in tokens if token and isinstance(token, str) and token.strip()]

        # Menggabungkan kembali kata-kata menjadi kalimat yang padu
        result = ' '.join(tokens)
        return result
    else:
        return None

def preprocess_text(text):
    if isinstance(text, list):
        # Jika input berupa daftar token, gabungkan menjadi string
        text = ' '.join(text)

    if text is not None:  # Tambahkan pengecekan untuk nilai None
        text = casefolding(text)
        text = cleaning(text)
        text = replace_abbreviations(text, indo_abbreviation)  # Menyertakan argumen 'abbreviation'
        text = replace_slank_words(text, indo_slank_words)    # Menyertakan argumen 'slank_words'
        text = remove_stop_words(text)
        text = stemming(text)
        text = ' '.join(tokenize(text))  # Ubah hasil tokenisasi menjadi string
        
        return text
    else:
        return None

# Fungsi untuk menyimpan dataset yang telah dilabeli
def save_labeled_dataset(dataset, dataset_name):
    dataset_path = f"uploaded_files/labeled_{dataset_name}"
    dataset.to_csv(dataset_path, index=False)
    return dataset_path     

# Fungsi untuk memuat dataset yang telah dilabeli
def load_labeled_dataset(dataset_path):
    return pd.read_csv(dataset_path)

# Variabel global untuk menyimpan dataset yang telah dilabeli
labeled_datasets = {}  # Gunakan dictionary untuk menyimpan multiple datasets

upload_dir = "uploaded_files"

if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# HALAMAN SCRAPPING DATA TWITTER
def get_tweet_data(card):
    """Extract data from tweet card"""
    try:
        username = card.find_element(By.XPATH, './/span').text
    except NoSuchElementException:
        username = ""

    try:
        handle = card.find_element(By.XPATH, './/span[contains(text(), "@")]').text
    except NoSuchElementException:
        handle = ""
    
    try:
        postdate = card.find_element(By.XPATH, './/time').get_attribute('datetime')
    except NoSuchElementException:
        postdate = ""
    
    try:
        text = card.find_element(By.XPATH, './/div[@data-testid="tweetText"]').text
    except NoSuchElementException:
        text = ""

    try:
        reply_cnt = card.find_element(By.XPATH, './/div[@data-testid="reply"]').text
    except NoSuchElementException:
        reply_cnt = "0"

    try:
        retweet_cnt = card.find_element(By.XPATH, './/div[@data-testid="retweet"]').text
    except NoSuchElementException:
        retweet_cnt = "0"

    try:
        like_cnt = card.find_element(By.XPATH, './/div[@data-testid="like"]').text
    except NoSuchElementException:
        like_cnt = "0"
    
    tweet = (username, handle, postdate, text, reply_cnt, retweet_cnt, like_cnt)
    return tweet

# Fungsi utama untuk scrapping
def start_scraping(Acc_Username, Acc_Password, Tweets_Query, dataset_name):
    driver = webdriver.Edge() # Gunakan driver yang sesuai untuk browser Anda
    driver.get('https://twitter.com/login')

    username_input = WebDriverWait(driver, timeout=60).until(
        EC.visibility_of_element_located((By.XPATH, '//input'))
    )
    username_input.send_keys(Acc_Username)
    next_btn = driver.find_element(By.XPATH, '//*[text()="Next"]')
    next_btn.click()

    # Tunggu untuk input password muncul
    password_input = WebDriverWait(driver, timeout=60).until(
        EC.visibility_of_element_located((By.XPATH, '//input[@name="password"]'))
    )
    password_input.send_keys(Acc_Password)

    login_btn = driver.find_element(By.XPATH, '//*[text()="Log in"]')
    login_btn.click()
            
    time.sleep(5)

    # Tunggu hingga tombol explore muncul dan klik
    explore_btn = WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, '//a[@href="/explore"]'))
    )
    explore_btn.click()

    # Tunggu hingga input eksplorasi muncul
    explore_input = WebDriverWait(driver, 60).until(
        EC.visibility_of_element_located((By.XPATH, '//input'))
    )
    explore_input.send_keys(Tweets_Query)
    explore_input.send_keys(Keys.ENTER)

    # Tunggu hingga tombol latest muncul dan klik
    latest_btn = WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, '//div[@data-testid][2]/div/div[2]'))
    )
    latest_btn.click()

    data = []
    tweet_ids = set()

    last_position = driver.execute_script("return window.pageYOffset;")
    scrolling = True
    wait = WebDriverWait(driver, 50)
    while scrolling:
        page_cards = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
        for card in page_cards[-15:]:
            tweet = get_tweet_data(card)
            if tweet:
                tweet_id = tweet
                if tweet_id not in tweet_ids:
                    tweet_ids.add(tweet_id)
                    data.append(tweet)
                        
        scroll_attempt = 0
        while True:
            # cek posisi scroll
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(3)
            curr_position = driver.execute_script("return window.pageYOffset;")
            if last_position == curr_position:
                scroll_attempt += 1
                            
                # akhir dari daerah scroll
                if scroll_attempt >= 3:
                    scrolling = False
                    break
                else:
                    time.sleep(2) # Coba gulir lagi
            else:
                last_position = curr_position
                break

    # Pastikan folder "Assets" ada
    folder_name = 'Assets'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Simpan dataset dengan nama yang ditentukan pengguna
    if dataset_name:
        file_name = f"{dataset_name}.csv"
        file_path = os.path.join(folder_name, file_name)

        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            header = ['UserName', 'Handle', 'Timestamp', 'Text', 'Comments', 'Likes', 'Retweets']
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    st.success(f"Scraping data Twitter selesai. Data disimpan di dalam folder 'Assets' dengan nama '{file_name}'.")
    driver.quit()

# Fungsi untuk menampilkan halaman sesuai session state
if st.session_state.page == 'Home':
    
    # Konten di halaman Home
    st.markdown("""
    <h3 style='text-align: center;'>ANALISIS SENTIMEN DALAM MENGETAHUI PERSEPSI PUBLIK TERHADAP REPRESENTASI KASUS LGBT DI MEDIA SOSIAL TWITTER MENGGUNAKAN PENDEKATAN NATURAL LANGUAGE PROCESSING DAN ALGORITMA NAIVE BAYES CLASSIFIER</h3>
    <h5 style='text-align: center;'></h5>
    """, unsafe_allow_html=True)

   # Daftar gambar
    images = ['image1.webp','image2.png', 'image3.png']

    # Tampilkan gambar secara berdampingan
    cols = st.columns(len(images))

    for col, img in zip(cols, images):
        with col:
            # Menggunakan st.image untuk menampilkan gambar dan memastikan teks caption ditampilkan dengan benar
            col.image(img, width=350, use_column_width=False, caption=f' {img}')
    
    st.markdown("""
    <h1 style='text-align: center;'></h1>           
    <h5 style='text-align: center;'>Pro atau kontra LGBT, pilihan di tangan lo! Gimana sih pandangan lo soal gerakan ini? Apapun keputusan lo, yang penting tetap saling menghargai dan menghormati, guys!</h5>
    <h1 style='text-align: center;'></h1>
    """, unsafe_allow_html=True)
    
    st.write ("""Instruksi Langkah-langkah Analisis Sentimen:
        
    1. Persiapan Dataset: Mulailah dengan menyiapkan dataset sentimen Twitter dalam format .csv. Jika belum punya, Anda bisa melakukan scraping data melalui menu Scraping.
        
    2. Upload Data: Gunakan menu Upload Data untuk mengimpor dataset sentimen yang akan dianalisis atau mengganti dataset yang ada dengan yang baru.

    3. Preprocessing Data: Bersihkan dan beri label data teks Anda melalui menu Preprocessing agar siap untuk analisis lebih lanjut.

    4. Analisis TF-IDF dan Sentimen: Lakukan analisis sentimen dengan algoritma Naive Bayes setelah melakukan proses TF-IDF di menu Result.

    5. Tampilkan Hasil Pengujian: Lihat hasil analisis sentimen Anda di menu Klasifikasi Naive Bayes. Di sini, Anda akan menemukan metrik akurasi, precision, dan recall, serta confusion matrix untuk evaluasi yang lebih mendalam.""")     

# HALAMAN DATASET
elif st.session_state.page == 'Dataset':
    st.title('Silakan upload dataset terlebih dahulu')
    data_file = st.file_uploader("Unggah file CSV", type=["csv"])


    if data_file is not None:
        # Membaca DataFrame dari file CSV yang diunggah
        df = pd.read_csv(data_file)
        
        # Menyimpan DataFrame ke dalam session state
        st.session_state.uploaded_data = df

    # Memeriksa apakah ada data yang diunggah di session state
    if 'uploaded_data' in st.session_state:
        st.title('Data yang Telah Diunggah')
        df = st.session_state.uploaded_data
        st.dataframe(df)

    st.markdown("""
    <h1 style='text-align: center;'></h1>           
    <h5 style='text-align: center;'>Lo belum punya DATASET? Lo bisa melakukan scraping data dibawah ini🙉</h5>
    <h1 style='text-align: center;'></h1>
    """, unsafe_allow_html=True)

    #PART SCRAPPING DATA
    st.title('Scrapping Data Twitter')  
    # Input untuk akun Twitter, topik pencarian, dan nama dataset
    Acc_Username = st.text_input("Masukkan username Twitter Anda:")
    Acc_Password = st.text_input("Masukkan password Twitter Anda:", type="password")
    Tweets_Query = st.text_input("Masukkan topik pencarian Twitter yang ingin di Scrapping:")
    dataset_name = st.text_input("Masukkan nama untuk dataset:")

    # Tombol untuk memulai scraping
    if st.button("Mulai Scraping"):
        # Periksa apakah akun Twitter, topik pencarian, dan nama dataset sudah dimasukkan
        if not Acc_Username or not Acc_Password or not Tweets_Query or not dataset_name:
            st.warning("Silakan masukkan username Twitter, password, topik pencarian, dan nama dataset terlebih dahulu.")
        else:
            start_scraping(Acc_Username, Acc_Password, Tweets_Query, dataset_name)

#HALAMAN PREPROCESSING DATA        
elif st.session_state.page == 'Preprocessing':
    st.title('Text Preprocessing')
    if 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        st.subheader('Data Sebelum Preprocessing')
        st.dataframe(df)
    else:
        st.warning('Mohon unggah dataset terlebih dahulu.')

    if st.button('Proses Preprocessing'):
        if menu_id == 'Preprocessing':
            dataset_name = st.session_state.uploaded_data
            df = df.drop(['UserName','Handle','Timestamp', 'Comments', 'Likes', 'Retweets'], axis=1)
            df['Case_Folding'] = df['Text'].fillna('').apply(lambda x: casefolding(x))
            df['Cleaning_Text'] = df['Case_Folding'].fillna('').apply(lambda x: cleaning(x))
            df['Normalisasi_Text'] = df['Cleaning_Text'].apply(replace_abbreviations, abbreviation=indo_abbreviation)
            df['Formalisasi_Text'] = df['Normalisasi_Text'].apply(replace_slank_words, slank_words=indo_slank_words)
            df['Remove_Text'] = df['Formalisasi_Text'].fillna('').apply(lambda x: remove_stop_words(x))
            df['Stemming_Text'] = df['Remove_Text'].fillna('').apply(lambda x: stemming(x))
            df['Tokenize_Text'] = df['Stemming_Text'].fillna('').apply(lambda x: tokenize(x))
            df['Processed_Text'] = df['Text'].fillna('').apply(lambda x: preprocess_text(x))
        
        st.session_state.preprocessed_data = df  # Simpan dataframe hasil preprocessing
        st.dataframe(df[['Text','Case_Folding','Cleaning_Text','Normalisasi_Text','Formalisasi_Text','Remove_Text','Stemming_Text','Tokenize_Text','Processed_Text']])

        label_column = 'Processed_Text'  # Mengganti label kolom dengan hasil preprocessing
        
        # Lakukan proses labelling dari kolom 'Hasil_Preprocessing'
        lexicon_positive = pd.read_excel('kamus_positive.xlsx')
        lexicon_positive_dict = {}
        for index, row in lexicon_positive.iterrows():
            if row[0] not in lexicon_positive_dict:
                lexicon_positive_dict[row[0]] = row[1]
        
        lexicon_negative = pd.read_excel('kamus_negative.xlsx')
        lexicon_negative_dict = {}
        for index, row in lexicon_negative.iterrows():
            if row[0] not in lexicon_negative_dict:
                lexicon_negative_dict[row[0]] = row[1]

        def sentiment_analysis_lexicon_indonesia(text):
            text = str(text)
            score = 0
            for word in text.split():
                if isinstance(word, str): 
                    if word.lower() in lexicon_positive_dict:
                        score += lexicon_positive_dict[word.lower()]
                    elif word.lower() in lexicon_negative_dict:
                        score += lexicon_negative_dict[word.lower()]

            sentimen = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
            return score, sentimen
        
        # Pastikan df sudah didefinisikan sebelumnya
        if 'df' not in locals():  # Cek apakah df sudah ada
            df = st.session_state.uploaded_data  # Inisialisasi df Anda di sini (misalnya, df = pd.read_csv('data.csv'))

        # Lakukan analisis sentimen
        results = df['Processed_Text'].apply(sentiment_analysis_lexicon_indonesia)
        results = list(zip(*results))
        df['Polarity Score'] = results[0]
        df['Labels'] = results[1]

        st.subheader('Data Setelah Preprocessing dan Labeling')
        st.dataframe(df[['Processed_Text', 'Polarity Score', 'Labels']])

        # Simpan dataframe hasil preprocessing dan labeling di session state
        st.session_state.preprocessed_data = df

        labeled_dataset_path = save_labeled_dataset(df, "labeled_dataset.csv")
        st.success('Proses preprocessing dan labeling selesai.')
        st.success(f'Dataset yang sudah diproses dan dilabeli disimpan di: {labeled_dataset_path}')

elif st.session_state.page == 'Predict':
    st.title('Hasil Klasifikasi Naive Bayes')

    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None

    if st.session_state.preprocessed_data is None:
        st.warning('Mohon lakukan preprocessing terlebih dahulu.')
    else:
        df = st.session_state.preprocessed_data  # Mendapatkan DataFrame dari session state

        if 'Processed_Text' in df.columns and 'Labels' in df.columns:
            st.subheader('Data Setelah Preprocessing dan Labeling')
            st.dataframe(df[['Processed_Text', 'Labels']])
        else:
            st.error("Kolom 'Processed_Text' atau 'Labels' tidak ditemukan dalam DataFrame. Mohon periksa kembali langkah preprocessing.")

    if st.button('Proses TF-IDF dan Naive Bayes'):
        non_empty_documents = df['Processed_Text'].fillna('')

        if non_empty_documents.empty:
            st.warning("Semua dokumen menjadi kosong setelah preprocessing. Periksa langkah-langkah preprocessing Anda.")
        else:
            df = df.dropna(subset=['Labels'])

            st.session_state.vectorizer = TfidfVectorizer()
            X_train = st.session_state.vectorizer.fit_transform(non_empty_documents)
            Y_train = df['Labels'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

            # Terapkan SMOTE untuk oversampling kelas minoritas
            smote = SMOTE(random_state=42)
            X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

            # Buat DataFrame dari matriks TF-IDF
            tfidf_df = pd.DataFrame(X_train_resampled.toarray(), columns=st.session_state.vectorizer.get_feature_names_out())

            st.subheader('Hasil TF-IDF')
            tfidf_df_100_words = tfidf_df.iloc[:, :100]
            st.dataframe(tfidf_df_100_words)
            st.write("===========================================================")

            X_train, X_test, Y_train, Y_test = train_test_split(X_train_resampled, Y_train_resampled, test_size=0.2, stratify=Y_train_resampled, random_state=42)
            
            # Class weight adjustment
            class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

            clfnb = MultinomialNB()
            clfnb.fit(X_train, Y_train)

            predict = clfnb.predict(X_test)
            predict_labels = ['Positif' if p == 1 else 'Negatif' if p == -1 else 'Netral' for p in predict]

            probability_values = clfnb.predict_proba(X_test)

            df_probability = pd.DataFrame(probability_values, columns=['Probability for Class Negatif', 'Probability for Class Netral', 'Probability for Class Positif'])

            st.write("Tabel Nilai Probabilitas:")
            st.write(df_probability)
            
            df_predictions = pd.DataFrame({'Data Ke': range(1, len(predict_labels) + 1), 'Prediksi': predict_labels})

            st.write("Hasil Prediksi dengan Naive Bayes:")
            st.write(df_predictions)
            
            positif_count = sum(predict == 1)
            negatif_count = sum(predict == -1)
            netral_count = sum(predict == 0)
            labels = ['Positif', 'Negatif', 'Netral']
            sizes = [positif_count, negatif_count, netral_count]
            colors = ['#BC7FCD','#FB9AD1','#FFCDEA']

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(sizes, labels=[f'{label}\n({size})' for label, size in zip(labels, sizes)], colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 8})
            ax.set_title('Jumlah Prediksi per Kelas', fontsize=10)
            st.pyplot(fig)
            
            prediksi_benar = (predict == Y_test).sum()
            prediksi_salah = (predict != Y_test).sum()

            st.write('Jumlah prediksi benar:', prediksi_benar)
            st.write('Jumlah prediksi salah:', prediksi_salah)

            accuracy = prediksi_benar / (prediksi_benar + prediksi_salah) * 100
            st.write('Akurasi pengujian:', accuracy, '%')
            
            st.write("===========================================================")
            accuracy = accuracy_score(predict, Y_test) * 100
            recall = recall_score(predict, Y_test, average='macro') * 100
            precision = precision_score(predict, Y_test, average='macro') * 100
            f1 = f1_score(predict, Y_test, average='macro') * 100

            skor_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score'],
                'Score': [accuracy, recall, precision, f1]
            })

            st.subheader('Skor Klasifikasi')
            st.dataframe(skor_df)
            
            st.write("===========================================================")
            st.subheader('Confusion Matrix')
            cm = confusion_matrix(Y_test, predict)

            plt.figure(figsize=(8, 5))
            sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues')
            plt.xlabel("Prediksi")
            plt.ylabel("True")
            plt.savefig("confusion_matrix.png")
            st.pyplot(plt.gcf())
            
            report = classification_report(Y_test, predict, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            st.write("===========================================================")
            st.subheader('Classification Report')
            st.dataframe(report_df)
            
            st.session_state.clfnb = clfnb
            st.session_state.tfidf_df = tfidf_df
            st.session_state.df_predictions = df_predictions
            st.session_state.fig = fig
            st.session_state.prediksi_benar = prediksi_benar
            st.session_state.prediksi_salah = prediksi_salah
            st.session_state.accuracy = accuracy
            st.session_state.skor_df = skor_df
            st.session_state.cm = cm
            st.session_state.report_df = report_df
