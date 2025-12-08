import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.title("🚗 Sistem CBR Prediksi Harga Mobil – CarDekho Dataset")

# ========== LOAD DATASET ==========
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
    return df

df = load_data()

# ========== PREPROCESSING UNTUK CBR ==========
df_cbr = df.copy()

# Label Encoding
label_cols = ["name", "fuel", "seller_type", "transmission", "owner"]
encoders = {}

for col in label_cols:
    enc = LabelEncoder()
    df_cbr[col] = enc.fit_transform(df_cbr[col])
    encoders[col] = enc

# Normalisasi fitur (CBR butuh jarak yang fair)
scaler = MinMaxScaler()

feature_cols = df_cbr.drop("selling_price", axis=1).columns
df_cbr_scaled = scaler.fit_transform(df_cbr[feature_cols])

# Dataset setelah scaling
df_scaled = pd.DataFrame(df_cbr_scaled, columns=feature_cols)
df_scaled["selling_price"] = df_cbr["selling_price"]

# ========== FORM INPUT ==========
st.subheader("🔧 Input Data Mobil:")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Tahun Mobil", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("KM Driven", min_value=100, max_value=500000, value=50000)
    seats = st.number_input("Jumlah Kursi", min_value=2, max_value=10, value=5)
    transmission = st.selectbox("Transmisi", df["transmission"].unique())

with col2:
    name = st.selectbox("Nama / Model Mobil", df["name"].unique())
    fuel = st.selectbox("Jenis Bahan Bakar", df["fuel"].unique())
    seller_type = st.selectbox("Tipe Penjual", df["seller_type"].unique())
    owner = st.selectbox("Status Kepemilikan", df["owner"].unique())

# Encode input
input_raw = {
    "name": encoders["name"].transform([name])[0],
    "year": year,
    "km_driven": km_driven,
    "fuel": encoders["fuel"].transform([fuel])[0],
    "seller_type": encoders["seller_type"].transform([seller_type])[0],
    "transmission": encoders["transmission"].transform([transmission])[0],
    "owner": encoders["owner"].transform([owner])[0],
    "seats": seats
}

input_df = pd.DataFrame([input_raw])
input_df = input_df[feature_cols]
input_scaled = scaler.transform(input_df)

# ========== CBR FUNCTION ==========
def cbr_predict(input_case, k=5):
    """
    CBR = Cari K kasus terdekat → Ambil harga rata-rata → Prediksi
    """

    # Data fitur tanpa selling_price
    case_base = df_scaled.drop("selling_price", axis=1).values

    # Hitung jarak euclidean
    distances = np.linalg.norm(case_base - input_case, axis=1)

    # Ambil K terdekat
    nearest_idx = distances.argsort()[:k]
    nearest_cases = df_scaled.iloc[nearest_idx]

    # Reuse → rata-rata harga
    predicted_price = nearest_cases["selling_price"].mean()

    return predicted_price, nearest_cases


KURS_INR_TO_IDR = 190  # konversi INR → IDR

# ========== PREDIKSI ==========
if st.button("Prediksi Harga (CBR)"):
    pred, neighbors = cbr_predict(input_scaled, k=5)
    pred_idr = pred * KURS_INR_TO_IDR

    st.subheader("💰 Hasil Prediksi Harga Berdasarkan Kasus Mirip (CBR)")
    st.success(f"Estimasi harga mobil: Rp {pred_idr:,.0f}")

    st.subheader("📌 Kasus yang Paling Mirip (Retrieve)")
    st.dataframe(neighbors)
