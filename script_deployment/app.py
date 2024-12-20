import streamlit as st
import numpy as np
import pickle
from sklearn.cluster import KMeans

with open('model/model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

st.title("Prediksi Cluster Menggunakan KMeans")

st.write("""
Aplikasi ini memungkinkan Anda untuk memprediksi cluster mana suatu data termasuk
berdasarkan 6 fitur yang dimasukkan. Silakan masukkan nilai untuk fitur berikut:
""")

energy = st.number_input('Energi (gram)', min_value=0.0, max_value=3000.0, step=0.1)
fat = st.number_input('Lemak (gram)', min_value=0.0, max_value=500.0, step=0.1)
saturated_fat = st.number_input('Lemak Jenuh (gram)', min_value=0.0, max_value=500.0, step=0.1)
carbohydrates = st.number_input('Karbohidrat (gram)', min_value=0.0, max_value=500.0, step=0.1)
sugars = st.number_input('Gula (gram)', min_value=0.0, max_value=500.0, step=0.1)
proteins = st.number_input('Protein (gram)', min_value=0.0, max_value=500.0, step=0.1)

input_data = np.array([[energy, fat, saturated_fat, carbohydrates, sugars, proteins]])

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

if st.button('Prediksi Cluster'):
    cluster_prediction = kmeans.predict(input_data)

    cluster_category = cluster_mapping.get(cluster_prediction[0], "Kategori tidak ditemukan")

    st.write(f"Data Anda termasuk dalam Cluster **{cluster_category}**")

    cluster_center = kmeans.cluster_centers_[cluster_prediction[0]]
    st.write(f"Koordinat pusat untuk cluster **{cluster_category}**:")
    st.write(cluster_center)
