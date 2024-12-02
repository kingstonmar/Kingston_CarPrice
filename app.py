import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Konfigurasi halaman
st.set_page_config(page_title="Estimasi Harga Mobil Bekas", layout="wide")

# Fungsi untuk memuat dataset
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    data = convert_object_columns(data)  # Konversi kolom bertipe object
    return data

# Fungsi untuk mengonversi kolom bertipe object menjadi tipe data yang sesuai
def convert_object_columns(data):
    for col in data.select_dtypes(include=['object']).columns:
        # Jika ada banyak kategori, ubah menjadi 'string'
        if data[col].nunique() < 10:  # Mengubah menjadi 'category' jika ada sedikit kategori
            data[col] = data[col].astype('category')
        else:
            data[col] = data[col].astype('string')  # Mengubah menjadi 'string' jika lebih banyak kategori
    return data

# Fungsi untuk preprocessing
def preprocess_data(data):
    data = data.dropna()  # Hapus nilai kosong
    data = pd.get_dummies(data, drop_first=True)  # Encoding
    return data

# Fungsi untuk membuat model
@st.cache_data
def train_model(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Sidebar menu
menu = st.sidebar.radio("Navigasi", ["Home", "EDA", "Prediction"])

# Home Page
if menu == "Home":
    st.title("Aplikasi Estimasi Harga Mobil Bekas")
    st.markdown(
        """
        Selamat datang di **Aplikasi Estimasi Harga Mobil Bekas**!  
        Aplikasi ini dirancang untuk membantu Anda dalam:
        
        - Mengunggah dataset untuk eksplorasi data.
        - Melakukan analisis eksplorasi data interaktif (EDA).
        - Memperkirakan harga mobil berdasarkan data yang Anda masukkan.
        """
    )
    
    # Tambahkan elemen interaktif
    st.header("Panduan Penggunaan Aplikasi")
    st.markdown(
        """
        1. Pilih **EDA** untuk eksplorasi dataset yang Anda unggah.
        2. Gunakan **Prediction** untuk memprediksi harga mobil bekas berdasarkan atribut dataset.
        """
    )

    # Form interaktif
    st.subheader("Coba Estimasi Cepat (Tanpa Dataset)")
    st.write("Masukkan beberapa informasi di bawah ini untuk estimasi cepat:")
    
    # Input dari pengguna
    brand = st.selectbox("Merk Mobil", ["Toyota", "Honda", "Suzuki", "Daihatsu", "Mitsubishi"])
    year = st.slider("Tahun Produksi", 2000, 2023, 2015)
    mileage = st.number_input("Kilometer Tempuh (dalam km)", value=50000, step=1000)
    transmission = st.selectbox("Transmisi", ["Manual", "Otomatis"])
    fuel_type = st.selectbox("Tipe Bahan Bakar", ["Bensin", "Diesel", "Hybrid", "Listrik"])

    if st.button("Estimasi Sekarang"):
        # Dummy prediksi (contoh)
        base_price = 100_000_000  # Harga dasar
        price = (
            base_price
            - (2023 - year) * 5_000_000
            - (mileage // 10_000) * 2_000_000
            + (20_000_000 if transmission == "Otomatis" else 0)
            + (10_000_000 if fuel_type in ["Hybrid", "Listrik"] else 0)
        )
        st.success(f"Estimasi Harga Mobil Anda: Rp {price:,.0f}")

# EDA Page
elif menu == "EDA":
    st.title("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"], key="eda")
    if uploaded_file is not None:  # Memastikan file diunggah
        try:
            data = load_data(uploaded_file)
            st.write("Dataset berhasil dimuat!")
            st.write("Tipe Data Kolom:")
            st.write(data.dtypes)  # Menampilkan tipe data kolom

            # Pilih kolom untuk distribusi
            selected_column = st.selectbox("Pilih Kolom untuk Visualisasi", data.columns)
            
            # Grafik interaktif menggunakan Plotly
            st.plotly_chart(
                px.histogram(data, x=selected_column, title=f"Distribusi {selected_column}", 
                             marginal="box", color_discrete_sequence=['teal'])
            )

            # Korelasi heatmap
            if st.checkbox("Tampilkan Korelasi Heatmap"):
                # Pastikan hanya kolom numerik yang diproses
                numeric_data = data.select_dtypes(include=['float64', 'int64'])

                if numeric_data.empty:
                    st.warning("Tidak ada kolom numerik untuk analisis korelasi.")
                else:
                    correlation_matrix = numeric_data.corr()
                    fig = px.imshow(
                        correlation_matrix, 
                        text_auto=True, 
                        title="Matriks Korelasi", 
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
    else:
        st.info("Silakan unggah file dataset dalam format CSV.")

# Prediction Page
elif menu == "Prediction":
    st.title("Estimasi Harga Mobil Bekas")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"], key="prediction")
    if uploaded_file:
        try:
            data = load_data(uploaded_file)
            st.write("Kolom dataset:", data.columns)
            
            # Pilih target
            target = st.selectbox("Pilih Kolom Target", data.columns)
            
            # Input Prediksi
            input_data = {}
            if st.checkbox("Input Nilai untuk Prediksi"):
                for col in data.drop(columns=[target]).columns:
                    value = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)
                    input_data[col] = value

            if st.button("Latih Model dan Prediksi"):
                data = preprocess_data(data)
                model, X_test, y_test = train_model(data, target)
                preds = model.predict(X_test)
                
                # Menampilkan hasil evaluasi
                st.success("Model berhasil dilatih!")
                st.write(f"R2 Score: {r2_score(y_test, preds):.2f}")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, preds):.2f}")
                
                # Prediksi berdasarkan input pengguna
                if input_data:
                    input_df = pd.DataFrame([input_data])
                    user_pred = model.predict(input_df)
                    st.write(f"Prediksi Harga Mobil Bekas: {user_pred[0]:,.2f}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.info("Silakan unggah file dataset Anda.")
