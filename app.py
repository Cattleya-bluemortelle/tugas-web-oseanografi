import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- KONFIGURASI NAMA APLIKASI ---
st.set_page_config(page_title="Ocean Dynamics Hub", layout="wide")
st.title("📊 Ocean Dynamics Hub")
st.markdown("*Platform Analisis Parameter Fisik Gelombang dan Arus Laut*")

# --- SIDEBAR: INPUT & SETTING ---
st.sidebar.header("⚙️ Pengaturan Analisis")
uploaded_file = st.sidebar.file_uploader("Unggah Data (CSV/Excel)", type=["csv", "xlsx"])
depth = st.sidebar.number_input("Kedalaman Laut (meter)", min_value=0.1, value=10.0, help="Diperlukan untuk klasifikasi jenis gelombang")

# --- FUNGSI FISIKA & FILTER ---
def butter_lowpass_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def calculate_physics(df, wave_col, d):
    g, rho = 9.81, 1025
    raw = df[wave_col].dropna().values
    clean = butter_lowpass_filter(raw)
    
    # Statistik Tinggi
    hs = 4 * np.std(clean)
    hrms = hs / np.sqrt(2)
    energy = 0.125 * rho * g * (hrms**2)
    
    # Estimasi Panjang & Jenis Gelombang (Teori Airy)
    T = 8.0 # Estimasi periode rata-rata
    L0 = (g * T**2) / (2 * np.pi)
    L = L0 * np.tanh(np.sqrt((4 * np.pi**2 * d) / (g * T)))
    ratio = d / L
    
    if ratio < 0.05: jenis = "Perairan Dangkal (Shallow Water)"
    elif ratio > 0.5: jenis = "Perairan Dalam (Deep Water)"
    else: jenis = "Perairan Transisi (Intermediate)"
    
    return raw, clean, hs, hrms, energy, jenis, L

# --- LOGIKA UTAMA ---
if uploaded_file:
    try:
        # Membaca file
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.sidebar.success(f"File '{uploaded_file.name}' dimuat.")
        
        # Deteksi Kolom Otomatis
        cols = [c.lower() for c in df.columns]
        wave_col = next((df.columns[i] for i, c in enumerate(cols) if any(k in c for k in ['elev', 'sea', 'z', 'water'])), None)
        u_col = next((df.columns[i] for i, c in enumerate(cols) if any(k in c for k in ['u_', 'vel_x', 'east', 'arus_u'])), None)
        v_col = next((df.columns[i] for i, c in enumerate(cols) if any(k in c for k in ['v_', 'vel_y', 'north', 'arus_v'])), None)

        if st.button("🚀 Jalankan Analisis Komprehensif"):
            tab_names = []
            if wave_col: tab_names.append("🌊 Gelombang")
            if u_col and v_col: tab_names.append("🏹 Arus")
            
            if not tab_names:
                st.error("Kolom data tidak terdeteksi. Gunakan nama kolom standar (u, v, elevasi).")
            else:
                tabs = st.tabs(tab_names)
                
                # --- ANALISIS GELOMBANG ---
                if wave_col:
                    with tabs[0]:
                        raw, clean, hs, hrms, energy, jenis, L = calculate_physics(df, wave_col, depth)
                        
                        st.subheader("Analisis Fisika Gelombang")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Significant Height (Hs)", f"{hs:.2f} m")
                        m2.metric("RMS Height (Hrms)", f"{hrms:.2f} m")
                        m3.metric("Rapat Energi", f"{energy:.1f} J/m²")
                        
                        st.info(f"**Klasifikasi Terdeteksi:** {jenis} (L ≈ {L:.2f} m)")
                        
                        fig1, ax1 = plt.subplots(figsize=(12, 4))
                        ax1.plot(raw, color='lightgray', alpha=0.5, label='Raw Signal')
                        ax1.plot(clean, color='#0077b6', linewidth=2, label='Filtered (Clean)')
                        ax1.set_ylabel("Elevation (m)")
                        ax1.set_title("Perbandingan Sinyal Elevasi Permukaan Laut")
                        ax1.legend()
                        st.pyplot(fig1)

                # --- ANALISIS ARUS ---
                if u_col and v_col:
                    idx = tab_names.index("🏹 Arus")
                    with tabs[idx]:
                        st.subheader("Distribusi Vektor Arus (Quiver Plot)")
                        u, v = df[u_col].values, df[v_col].values
                        mag = np.sqrt(u**2 + v**2)
                        
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        # Membuat koordinat sederhana jika tidak ada kolom X, Y
                        x = np.linspace(0, 10, len(u))
                        y = np.sin(x) # Hanya untuk sebaran visual
                        
                        q = ax2.quiver(x, y, u, v, mag, cmap='YlGnBu')
                        plt.colorbar(q, label='Velocity (m/s)')
                        ax2.set_title("Vektor Kecepatan dan Arah Arus")
                        st.pyplot(fig2)
                        st.success(f"Kecepatan Arus Rata-rata: {np.mean(mag):.3f} m/s")

    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
else:
    st.info("👋 Selamat Datang di Ocean Dynamics Hub. Silakan unggah data Anda di sidebar.")
