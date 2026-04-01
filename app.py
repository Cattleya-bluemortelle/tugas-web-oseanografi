import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import io

# --- KONFIGURASI APLIKASI ---
st.set_page_config(page_title="Ocean Dynamics Hub", layout="wide")
st.title("📊 Ocean Dynamics Hub")
st.markdown("*Analisis Dinamika Laut Fleksibel (Support Format JOGOS & Standar)*")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah File Data", type=["csv", "xlsx"])
depth = st.sidebar.number_input("Kedalaman Laut (m)", min_value=0.1, value=10.0)

def butter_lowpass_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

if uploaded_file is not None:
    try:
        # Trik untuk handle file JOGOS yang punya banyak header di atas
        file_bytes = uploaded_file.read()
        raw_text = file_bytes.decode('utf-8', errors='ignore')
        
        # Cari di baris mana data angka dimulai (biasanya setelah baris [Data])
        start_line = 0
        lines = raw_text.splitlines()
        for i, line in enumerate(lines):
            if "[Data]" in line or "Date" in line:
                start_line = i + 1
                break
        
        # Baca ulang file dari baris data tersebut
        uploaded_file.seek(0)
        if uploaded_file.name.endswith('.csv'):
            # Coba baca dengan header otomatis, jika gagal baru manual
            df = pd.read_csv(io.StringIO(raw_text), skiprows=start_line)
            if df.shape[1] < 2: # Jika gagal deteksi kolom, baca tanpa header
                df = pd.read_csv(io.StringIO(raw_text), skiprows=start_line, header=None)
        else:
            df = pd.read_excel(uploaded_file)

        # --- PEMETAAN KOLOM FLEKSIBEL (SMART MAPPING) ---
        # Membersihkan nama kolom agar mudah dicari
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Cari kolom berdasarkan kata kunci
        u_col = next((c for c in df.columns if any(k in c for k in ['u', 'vel_x', 'east', 'arus_u'])), None)
        v_col = next((c for c in df.columns if any(k in c for k in ['v', 'vel_y', 'north', 'arus_v'])), None)
        wave_col = next((c for c in df.columns if any(k in c for k in ['elev', 'sea', 'z', 'water', 'depth', 'pres'])), None)

        st.success(f"File '{uploaded_file.name}' berhasil dianalisis.")
        
        with st.expander("Lihat Pratinjau Kolom Terdeteksi"):
            st.write(f"Kolom Arus: **{u_col}**, **{v_col}**")
            st.write(f"Kolom Gelombang: **{wave_col}**")
            st.dataframe(df.head())

        if st.button("🚀 Jalankan Analisis Komprehensif"):
            tab_names = []
            if wave_col: tab_names.append("🌊 Gelombang")
            if u_col and v_col: tab_names.append("🏹 Arus")
            
            if not tab_names:
                st.error("Maaf, kolom data tidak bisa dikenali secara otomatis. Pastikan nama kolom mengandung kata 'U', 'V', atau 'Elevasi'.")
            else:
                tabs = st.tabs(tab_names)
                
                # --- ANALISIS GELOMBANG ---
                if wave_col:
                    with tabs[0]:
                        st.subheader("Parameter Fisika Gelombang")
                        raw_data = pd.to_numeric(df[wave_col], errors='coerce').dropna().values
                        
                        if len(raw_data) > 0:
                            clean_data = butter_lowpass_filter(raw_data)
                            
                            # Statistik Ilmiah
                            hs = 4 * np.std(clean_data)
                            hrms = hs / np.sqrt(2)
                            energy = 0.125 * 1025 * 9.81 * (hrms**2)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Sig. Height (Hs)", f"{hs:.2f} m")
                            col2.metric("RMS Height (Hrms)", f"{hrms:.2f} m")
                            col3.metric("Energy Density", f"{energy:.1f} J/m²")
                            
                            fig1, ax1 = plt.subplots(figsize=(12, 4))
                            ax1.plot(raw_data, color='lightgray', alpha=0.5, label='Raw Signal')
                            ax1.plot(clean_data, color='#0077b6', linewidth=2, label='Filtered Wave')
                            ax1.set_ylabel("Elevation (m)")
                            ax1.legend()
                            st.pyplot(fig1)
                        else:
                            st.warning("Data gelombang tidak valid atau kosong.")

                # --- ANALISIS ARUS ---
                if u_col and v_col:
                    idx = tab_names.index("🏹 Arus")
                    with tabs[idx]:
                        st.subheader("Visualisasi Vektor Arus (Quiver Plot)")
                        u = pd.to_numeric(df[u_col], errors='coerce').dropna().values
                        v = pd.to_numeric(df[v_col], errors='coerce').dropna().values
                        
                        # Potong agar panjangnya sama
                        min_len = min(len(u), len(v))
                        u, v = u[:min_len], v[:min_len]
                        
                        # Konversi JOGOS (cm/s ke m/s) jika angka terlalu besar
                        if np.mean(np.abs(u)) > 10: 
                            u, v = u/100, v/100
                            
                        mag = np.sqrt(u**2 + v**2)
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        # Tampilan sebaran titik data
                        x = np.linspace(0, 10, len(u))
                        y = np.zeros(len(u))
                        
                        skip = max(1, len(u)//50) # Ambil 50 titik saja biar tidak numpuk
                        q = ax2.quiver(x[::skip], y[::skip], u[::skip], v[::skip], mag[::skip], cmap='YlGnBu')
                        plt.colorbar(q, label='Velocity (m/s)')
                        ax2.set_title("Arah dan Kecepatan Arus")
                        st.pyplot(fig2)
                        st.info(f"Kecepatan Rata-rata: {np.mean(mag):.3f} m/s")

    except Exception as e:
        st.error(f"Terjadi kendala pembacaan file: {e}")
else:
    st.info("Silakan unggah file data oseanografi Anda.")
