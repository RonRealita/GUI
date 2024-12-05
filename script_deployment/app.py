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
energy = st.number_input('Energi (gram)', min_value=0.0, max_value=3000.0, step=0.1)
fat = st.number_input('Lemak (gram)', min_value=0.0, max_value=500.0, step=0.1)
saturated_fat = st.number_input('Lemak Jenuh (gram)', min_value=0.0, max_value=500.0, step=0.1)
carbohydrates = st.number_input('Karbohidrat (gram)', min_value=0.0, max_value=500.0, step=0.1)
sugars = st.number_input('Gula (gram)', min_value=0.0, max_value=500.0, step=0.1)
proteins = st.number_input('Protein (gram)', min_value=0.0, max_value=500.0, step=0.1)

# Membuat array input untuk prediksi (menggabungkan semua input ke dalam satu array)
input_data = np.array([[energy, fat, saturated_fat, carbohydrates, sugars, proteins]])

# Mapping cluster ke kategori deskriptif
cluster_mapping = {
    1: "Sayuran",
    2: "Produk Olahan Susu",
    3: "Produk Olahan Hewani",
    4: "Daging",
    5: "Jus dan Buah",
    6: "Saus dan Bumbu dapur",
    7: "Keju",
    8: "Minuman"
}

# Menampilkan prediksi saat pengguna mengklik tombol
if st.button('Prediksi Cluster'):
    # Menggunakan model KMeans untuk memprediksi cluster dari input data
    cluster_prediction = kmeans.predict(input_data)

    # Mencari kategori berdasarkan hasil prediksi cluster
    cluster_category = cluster_mapping.get(cluster_prediction[0], "Kategori tidak ditemukan")

    # Menampilkan hasil prediksi
    st.write(f"Data Anda termasuk dalam Cluster **{cluster_category}**")

    # Menampilkan informasi lebih lanjut tentang pusat cluster (opsional)
    cluster_center = kmeans.cluster_centers_[cluster_prediction[0]]
    st.write(f"Koordinat pusat untuk cluster **{cluster_category}**:")
    st.write(cluster_center)
