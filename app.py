import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import signal
from sklearn.linear_model import LinearRegression

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Ocean Dynamics Hub",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS untuk tampilan profesional
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    div.stButton > button:first-child {
        background-color: #00f2ff; color: #0e1117; font-weight: bold; border-radius: 20px; width: 100%;
        border: none; padding: 10px 20px;
    }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI SMART CLEANER (LOGIKA OTOMATIS) ---
def smart_cleaner(data):
    # Konversi ke numerik & tangani data kosong (Interpolasi)
    series = pd.to_numeric(data, errors='coerce')
    gap_count = series.isnull().sum()
    
    # Interpolasi dan pembersihan data kosong versi Pandas Terbaru
    series = series.interpolate(method='linear').bfill().ffill()
    
    # Analisis Statistik untuk Menentukan Metode
    std_dev = series.std()
    diff_abs = np.abs(series.diff().fillna(0))
    mean_val = series.mean()
    
    # Kasus A: Banyak Spikes/Noise Tajam (TSS, Salinitas, Turbiditas kotor)
    if diff_abs.max() > (3.5 * std_dev):
        window = max(3, len(series) // 50)
        processed = series.rolling(window=window, center=True).mean().bfill().ffill()
        method, color = "Moving Average (Smoothing)", "#00ff88"
        desc = f"Sistem mendeteksi lonjakan tajam (noise). Menggunakan perataan jendela {window} poin."
    
    # Kasus B: Sinyal Osilasi (Pasut, Gelombang, Tekanan)
    elif std_dev > 0.05 and len(series) > 50:
        # Low-pass Butterworth filter
        b, a = signal.butter(2, 0.1, btype='low')
        processed = signal.filtfilt(b, a, series)
        method, color = "Low-Pass Butterworth Filter", "#00d1ff"
        desc = "Sistem mendeteksi pola osilasi. Membuang riak frekuensi tinggi agar tren utama terlihat."
    
    # Kasus C: Tren Linier/Drift (Suhu, Penurunan Muka Tanah)
    else:
        x = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression().fit(x, series.values)
        processed = model.predict(x)
        method, color = "Linear Curve Fitting (Trend)", "#ffea00"
        desc = "Sistem mendeteksi tren perubahan jangka panjang tanpa osilasi besar."

    return series, processed, method, color, desc, gap_count

# --- TAMPILAN UTAMA ---
st.title("🌊 Ocean Dynamics Hub")
st.markdown("*Platform Analisis Cerdas untuk Parameter Oseanografi & Lingkungan Laut*")
st.write("---")

# --- SIDEBAR & UPLOAD ---
with st.sidebar:
    st.header("📂 Pusat Data")
    uploaded_file = st.file_uploader("Unggah CSV atau Excel", type=['csv', 'xlsx'])
    st.markdown("---")
    st.info("💡 **Tips:** Sistem ini akan otomatis mendeteksi baris data dan memilih metode pembersihan terbaik.")

if uploaded_file:
    try:
        # Membaca file dengan deteksi baris data otomatis (Header-Agnostic)
        if uploaded_file.name.endswith('.csv'):
            raw_lines = uploaded_file.read().decode('utf-8', errors='ignore').splitlines()
            # Mencari kata kunci [Data] atau Date untuk melewati header alat
            start_idx = next((i for i, line in enumerate(raw_lines) if "[Data]" in line or "Date" in line), -1)
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=start_idx+1 if start_idx != -1 else 0)
        else:
            df = pd.read_excel(uploaded_file)

        # Bersihkan nama kolom
        df.columns = [str(c).strip() for c in df.columns]

        # 1. PREVIEW DATA
        with st.expander("🔍 Pratinjau Data Mentah", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Terdeteksi {df.shape[0]} baris data.")

        # 2. KONFIGURASI VARIABEL
        st.subheader("⚙️ Pengaturan Grafik")
        col_x, col_y, col_btn = st.columns([2, 2, 1])
        
        with col_x:
            x_axis = st.selectbox("Pilih Sumbu X (Waktu/Index):", df.columns)
        with col_y:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            y_axis = st.selectbox("Pilih Parameter (Sumbu Y):", numeric_cols if numeric_cols else df.columns)
        with col_btn:
            st.write(" ") # Spacer
            run_btn = st.button("🚀 ANALISIS SEKARANG")

        if run_btn:
            # Jalankan Smart Cleaner
            raw_data, clean_data, method_name, line_color, info_text, gaps = smart_cleaner(df[y_axis])
            
            # 3. VISUALISASI INTERAKTIF
            fig = go.Figure()
            # Data Mentah (Abu-abu Transparan)
            fig.add_trace(go.Scatter(x=df[x_axis], y=raw_data, name="Data Mentah (Noise)", 
                                     line=dict(color='rgba(150,150,150,0.3)', width=1)))
            # Data Bersih (Warna sesuai metode)
            fig.add_trace(go.Scatter(x=df[x_axis], y=clean_data, name=f"Cleaned ({method_name})", 
                                     line=dict(color=line_color, width=3)))
            
            fig.update_layout(
                title=f"Perbandingan Data: {y_axis} vs {x_axis}",
                xaxis_title=x_axis, yaxis_title=y_axis,
                template="plotly_dark", height=500, hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 4. LAPORAN HASIL
            st.subheader("📋 Laporan Analisis")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Metode Terpilih", method_name)
            m2.metric("Nilai Rata-rata", f"{np.mean(clean_data):.2f}")
            m3.metric("Data Kosong Diisi", f"{gaps} titik")
            m4.metric("Stabilitas Sinyal", "Stabil" if gaps < (len(df)*0.1) else "Banyak Gap")
            
            st.success(f"**Kenapa metode ini dipilih?** {info_text}")

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan: {str(e)}")
else:
    st.info("Silakan unggah file data oseanografi Anda untuk memulai pemrosesan.")
    # Opsional: Tampilkan gambar ilustrasi jika belum ada data
