import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Konfigurasi Halaman
st.set_page_config(page_title="Oceanic Data Refinery", layout="wide")

st.title("🌊 Oceanic Data Refinery")
st.markdown("Aplikasi web ini memproses data mentah oseanografi menjadi data siap analisis menggunakan metode statistik dan pembersihan sinyal.")

# --- SIDEBAR ---
st.sidebar.header("Opsi Pengolahan Data")
uploaded_file = st.sidebar.file_uploader("Upload file data_arus.csv", type=["csv"])

# Jika belum ada file, buat data simulasi agar dosen bisa langsung lihat demo
if uploaded_file is not None:
    # Membaca file dengan desimal titik (.) sesuai settingan Excelmu
    df = pd.read_csv(uploaded_file, decimal='.')
else:
    st.sidebar.warning("Silakan upload data_arus.csv. Menampilkan data demo...")
    t = np.linspace(0, 50, 500)
    elevasi_noisy = np.sin(0.5 * t) + np.random.normal(0, 0.3, 500)
    df = pd.DataFrame({
        'waktu': t,
        'elevasi': elevasi_noisy,
        'x': np.repeat(np.arange(5), 5),
        'y': np.tile(np.arange(5), 5),
        'u': np.random.uniform(-0.5, 0.5, 25),
        'v': np.random.uniform(-0.5, 0.5, 25)
    })

tabs = st.tabs(["📊 Filter Gelombang", "🏹 Vektor Arus (Quiver)"])

# --- TAB 1: FILTERING ---
with tabs[0]:
    st.header("Metode Pembersihan Noise (Butterworth Filter)")
    st.write("Data mentah dari sensor seringkali memiliki gangguan (noise). Kita menggunakan Low-pass filter untuk mendapatkan sinyal murni.")
    
    cutoff = st.slider("Atur Kehalusan (Cutoff Frequency)", 0.01, 0.5, 0.1)
    
    # Proses filtering
    b, a = butter(3, cutoff)
    df['elevasi_filtered'] = filtfilt(b, a, df['elevasi'])
    
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df['elevasi'], label="Data Mentah (Noisy)", color='lightgray', alpha=0.6)
    ax1.plot(df['elevasi_filtered'], label="Hasil Filter (Clean)", color='blue', linewidth=2)
    ax1.set_title("Analisis Perbandingan Sinyal")
    ax1.legend()
    st.pyplot(fig1)

# --- TAB 2: QUIVER ---
with tabs[1]:
    st.header("Visualisasi Vektor Arus Spasial")
    st.write("Quiver plot menunjukkan arah dan kekuatan arus di berbagai koordinat.")
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    magnitude = np.sqrt(df['u']**2 + df['v']**2)
    q = ax2.quiver(df['x'], df['y'], df['u'], df['v'], magnitude, cmap='jet')
    plt.colorbar(q, label='Kecepatan Arus (m/s)')
    ax2.set_xlabel("Koordinat X")
    ax2.set_ylabel("Koordinat Y")
    st.pyplot(fig2)

st.markdown("---")
st.caption("Dikembangkan oleh: cattleya-bluemortelle | Matkul Analisis Data Oseanografi")
