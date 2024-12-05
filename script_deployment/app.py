import streamlit as st
import numpy as np
import pickle
from sklearn.cluster import KMeans

# Memuat model KMeans yang sudah dilatih
with open('model/model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

# Judul aplikasi Streamlit
st.title("Prediksi Cluster Menggunakan KMeans")

# Deskripsi aplikasi
st.write("""
Aplikasi ini memungkinkan Anda untuk memprediksi cluster mana suatu data termasuk
berdasarkan 6 fitur yang dimasukkan. Silakan masukkan nilai untuk fitur berikut:
""")

# Input pengguna untuk 6 fitur (disesuaikan dengan dataset yang Anda gunakan)
energy_100g = st.number_input('Energy (100g)', min_value=0.0, max_value=3000.0, step=0.1)
fat_100g = st.number_input('Fat (100g)', min_value=0.0, max_value=500.0, step=0.1)
saturated_fat_100g = st.number_input('Saturated Fat (100g)', min_value=0.0, max_value=500.0, step=0.1)
carbohydrates_100g = st.number_input('Carbohydrates (100g)', min_value=0.0, max_value=500.0, step=0.1)
sugars_100g = st.number_input('Sugars (100g)', min_value=0.0, max_value=500.0, step=0.1)
proteins_100g = st.number_input('Proteins (100g)', min_value=0.0, max_value=500.0, step=0.1)

# Membuat array input untuk prediksi (menggabungkan semua input ke dalam satu array)
input_data = np.array([[energy_100g, fat_100g, saturated_fat_100g, carbohydrates_100g, sugars_100g, proteins_100g]])

# Menampilkan prediksi saat pengguna mengklik tombol
if st.button('Prediksi Cluster'):
    # Menggunakan model KMeans untuk memprediksi cluster dari input data
    cluster_prediction = kmeans.predict(input_data)

    # Menampilkan hasil prediksi
    st.write(f"Data Anda termasuk dalam Cluster {cluster_prediction[0]}")

    # Menampilkan informasi lebih lanjut tentang pusat cluster (opsional)
    cluster_center = kmeans.cluster_centers_[cluster_prediction[0]]
    st.write(f"Koordinat pusat untuk cluster {cluster_prediction[0]}:")
    st.write(cluster_center)
    
    # Menampilkan informasi tambahan atau grafik jika diperlukan
    st.write("Anda dapat mengimpor dataset dan melihat distribusi cluster di visualisasi.")
