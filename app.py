import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import signal
from sklearn.linear_model import LinearRegression
import io

# --- KUSTOMISASI TAMPILAN (CSS) ---
st.set_page_config(page_title="Ocean Dynamics Hub", layout="wide", page_icon="🌊")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    div.stButton > button:first-child {
        background-color: #00f2ff; color: #0e1117; font-weight: bold; border-radius: 20px; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI SMART CLEANER (LOGIKA NUMERIK) ---
def smart_cleaner(data):
    # 1. Interpolasi Otomatis (Mengisi data bolong)
    series = pd.to_numeric(data, errors='coerce')
    gap_count = series.isnull().sum()
    series = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    # 2. Analisis Statistik Karakteristik Sinyal
    std_dev = series.std()
    diff_abs = np.abs(series.diff().fillna(0))
    
    # A. Jika banyak lonjakan tajam (Spikes) -> Moving Average
    if diff_abs.max() > (3.5 * std_dev):
        window = max(3, len(series) // 60)
        processed = series.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        method, color = "Moving Average (Smoothing)", "#00ff88"
        desc = f"Mendeteksi noise tajam (spikes). Menggunakan perataan jendela {window} poin."
    
    # B. Jika sinyal berosilasi (Pasut/Gelombang) -> Low-Pass Filter
    elif std_dev > 0.1 and len(series) > 50:
        b, a = signal.butter(2, 0.08, btype='low')
        processed = signal.filtfilt(b, a, series)
        method, color = "Low-Pass Butterworth", "#00d1ff"
        desc = "Mendeteksi pola osilasi. Membuang riak frekuensi tinggi (noise alat)."
    
    # C. Jika data memiliki tren kuat -> Curve Fitting (Regresi)
    else:
        x = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression().fit(x, series.values)
        processed = model.predict(x)
        method, color = "Linear Curve Fitting", "#ffea00"
        desc = "Mendeteksi tren perubahan jangka panjang. Menghilangkan fluktuasi kecil."

    return series, processed, method, color, desc, gap_count

# --- HEADER ---
st.title("🌊 Ocean Dynamics Hub")
st.markdown("### *Advanced Oceanographic Signal Processing & Environment Analytics*")
st.write("---")

# --- SIDEBAR & DATA LOADING ---
with st.sidebar:
    st.header("📂 Data Explorer")
    uploaded_file = st.file_uploader("Upload File (CSV/Excel)", type=['csv', 'xlsx'])
    st.markdown("---")
    st.write("**Petunjuk:**")
    st.caption("1. Unggah data mentah hasil survei.")
    st.caption("2. Lihat pratinjau data di kolom tengah.")
    st.caption("3. Pilih variabel yang ingin dianalisis.")
    st.caption("4. Klik tombol proses untuk pembersihan otomatis.")

if uploaded_file:
    try:
        # Cerdas membaca header (Universal untuk JOGOS/Manual)
        if uploaded_file.name.endswith('.csv'):
            raw = uploaded_file.read().decode('utf-8', errors='ignore').splitlines()
            start_row = next((i for i, line in enumerate(raw) if "[Data]" in line or "Date" in line), -1)
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=start_row+1 if start_row != -1 else 0)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = [str(c).strip() for c in df.columns]

        # --- PANEL PREVIEW ---
        with st.expander("🔍 Lihat Pratinjau Isi File (Preview)", expanded=True):
            st.dataframe(df.head(8), use_container_width=True)
            st.info(f"📊 Dataset: {df.shape[0]} Baris | {df.shape[1]} Kolom")

        # --- KONFIGURASI ANALISIS ---
        st.subheader("⚙️ Konfigurasi Analisis")
        c1, c2, c3 = st.columns([2, 2, 1])
        
        with c1:
            x_col = st.selectbox("Pilih Sumbu X (Waktu/Index):", df.columns)
        with c2:
            # Otomatis hanya kolom angka
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            y_col = st.selectbox("Pilih Parameter (Sumbu Y):", numeric_cols if numeric_cols else df.columns)
        with c3:
            st.write(" ") # Spacer
            process_btn = st.button("🚀 PROSES DATA")

        if process_btn:
            # Jalankan Mesin Otomatis
            raw_v, clean_v, method, color, desc, gaps = smart_cleaner(df[y_col])
            
            # --- VISUALISASI UTAMA ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[x_col], y=raw_v, name="Raw Noise", 
                                     line=dict(color='rgba(150,150,150,0.3)', width=1)))
            fig.add_trace(go.Scatter(x=df[x_col], y=clean_v, name=f"Cleaned ({method})", 
                                     line=dict(color=color, width=3)))
            
            fig.update_layout(
                title=f"Hasil Analisis Otomatis: {y_col}",
                template="plotly_dark", height=500,
                xaxis_title=x_col, yaxis_title=y_col,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- STATISTIK & LAPORAN ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Metode Terpilih", method)
            m2.metric("Rata-rata", f"{np.mean(clean_v):.2f}")
            m3.metric("Nilai Maks", f"{np.max(clean_v):.2f}")
            m4.metric("Data Kosong", f"{gaps} titik")
            
            st.success(f"**Analisis Sistem:** {desc}")

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan pembacaan: {e}")
else:
    # Tampilan saat belum ada file
    st.info("👋 Selamat Datang! Silakan unggah file CSV atau Excel di panel kiri untuk memulai.")
    st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_column_width=True)
