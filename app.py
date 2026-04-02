import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from fuzzywuzzy import fuzz
import warnings

# Mengabaikan warning agar tampilan tetap bersih
warnings.filterwarnings('ignore')

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="HydroData Ultimate QC",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("# 🌊 **HydroData Cleaner & Visualizer (Expert Edition)**")
st.markdown("*Professional Quality Control untuk Water Level, Salinitas, & Suhu*")
st.markdown("---")

class HydroDataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.detected_columns = self._smart_column_detection()
    
    def _smart_column_detection(self):
        """Mendeteksi tipe data secara otomatis menggunakan Fuzzy Logic"""
        keywords = {
            'Salinitas': ['sal', 'salinity', 'psu', 'pss', 'salin'],
            'Suhu': ['temp', 'temperature', 'suhu', 't', 'degc'],
            'Water Level': ['water_level', 'wl', 'elev', 'elevation', 'height', 'z', 'pasut', 'depth'],
            'Waktu': ['time', 'datetime', 'date', 'waktu', 'tanggal']
        }
        mapping = {}
        for col in self.df.columns:
            col_lower = str(col).lower()
            for data_type, words in keywords.items():
                if any(fuzz.partial_ratio(col_lower, w) > 75 or w in col_lower for w in words):
                    mapping[col] = data_type
                    break
        return mapping

    def apply_hard_qc(self, data, col_type):
        """Langkah krusial: Membuang angka error sensor (99.99 dll)"""
        data_clean = data.copy()
        
        # Batas Logis Oseanografi (Hard Limits)
        limits = {
            'Salinitas': (0, 45),
            'Suhu': (0, 40),
            'Water Level': (-5, 20)
        }
        
        low, high = limits.get(col_type, (np.nanmin(data), np.nanmax(data)))
        
        # 1. Hard Filter: Buang angka error ekstrem
        data_clean[(data_clean > high) | (data_clean < low)] = np.nan
        
        # 2. Statistical Filter: Buang spike tajam (IQR)
        q1 = np.nanpercentile(data_clean, 25)
        q3 = np.nanpercentile(data_clean, 75)
        iqr = q3 - q1
        data_clean[(data_clean < (q1 - 2*iqr)) | (data_clean > (q3 + 2*iqr))] = np.nan
        
        # 3. Interpolasi Linear (Mengisi gap kosong agar tren terjaga)
        return pd.Series(data_clean).interpolate(method='linear').bfill().ffill().values

    def moving_average(self, data, window=5):
        return pd.Series(data).rolling(window=window, center=True).mean().bfill().ffill().values

    def auto_noise_removal(self, data, col_type):
        """Sistem otomatis: QC dulu baru Smoothing"""
        # LANGKAH 1: WAJIB QC Outlier dulu agar 99.99 tidak merusak rata-rata
        qc_data = self.apply_hard_qc(data, col_type)
        
        # LANGKAH 2: Cek noise (getaran kecil)
        noise_level = np.nanstd(np.diff(qc_data))
        
        # LANGKAH 3: Hanya smoothing jika data kasar
        if noise_level > 0.05:
            cleaned = self.moving_average(qc_data, window=3)
            method = "Strict QC + Light Smoothing"
        else:
            cleaned = qc_data
            method = "Strict QC (Original Preserved)"
            
        return cleaned, method

# 3. FUNGSI VISUALISASI UTAMA
def create_expert_dashboard(df, selected_col, processor):
    col_type = processor.detected_columns.get(selected_col, "Data")
    
    # DATA ASLI (Raw - Tanpa Ubahan)
    raw_data = df[selected_col].values 
    
    # DATA QC (Hasil Olahan)
    cleaned_data, method_name = processor.auto_noise_removal(raw_data, col_type)
    
    time_idx = np.arange(len(raw_data))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '1. RAW DATA (Original File)', 
            '2. CLEANED DATA (QC Result)', 
            '3. ZOOM COMPARISON (150 Pts)', 
            '4. NOISE/ERROR REMOVED'
        ),
        vertical_spacing=0.15, horizontal_spacing=0.1
    )

    # Plot 1: Raw (Harus kelihatan spike ke 99.99)
    fig.add_trace(go.Scatter(x=time_idx, y=raw_data, name='Raw (Original)', line=dict(color='red', width=1)), row=1, col=1)

    # Plot 2: Cleaned (Harus stabil di angka 30-an)
    fig.add_trace(go.Scatter(x=time_idx, y=cleaned_data, name='Cleaned (QC)', line=dict(color='green', width=1.5)), row=1, col=2)

    # Plot 3: Zoom (Overlay Raw vs QC)
    zoom_n = min(len(raw_data), 150)
    fig.add_trace(go.Scatter(x=time_idx[:zoom_n], y=raw_data[:zoom_n], name='Raw Zoom', line=dict(color='red', width=1, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_idx[:zoom_n], y=cleaned_data[:zoom_n], name='QC Zoom', line=dict(color='green', width=2)), row=2, col=1)

    # Plot 4: Residual
    residual = raw_data - cleaned_data
    fig.add_trace(go.Scatter(x=time_idx, y=residual, name='Residual/Noise', line=dict(color='purple')), row=2, col=2)

    fig.update_layout(height=850, template="plotly_white", title=f"<b>QC Dashboard: {selected_col}</b> | Method: {method_name}")
    return fig, cleaned_data, method_name

# 4. MAIN APP
def main():
    st.sidebar.header("📁 **Data Input**")
    uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx'])

    if not uploaded_file:
        st.info("👆 Silakan upload file data oseanografi di sidebar.")
        st.stop()

    # Load Data
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        
        processor = HydroDataProcessor(df_numeric)
        st.sidebar.success(f"✅ Loaded {len(df_numeric)} baris data.")
        
        selected_col = st.sidebar.selectbox("Pilih Parameter Oseanografi:", df_numeric.columns)
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 **Expert Visualisasi**", "🧹 **Auto Clean & Download**", "⚙️ **Manual QC Tools**"])

    with tab1:
        st.header("📈 **Analisis Perbandingan QC**")
        if selected_col:
            fig, cleaned_res, method = create_expert_dashboard(df_numeric, selected_col, processor)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw Max", f"{np.nanmax(df_numeric[selected_col]):.2f}")
            c2.metric("Cleaned Mean", f"{np.mean(cleaned_res):.2f}")
            c3.metric("Status QC", "Passed" if np.mean(cleaned_res) < 45 else "Warning")

    with tab2:
        st.header("🧹 **Export Data Bersih**")
        if st.button("🚀 Proses Semua Data & Siapkan Download", type="primary"):
            df_final = df.copy()
            col_type = processor.detected_columns.get(selected_col, "Data")
            cleaned_final, _ = processor.auto_noise_removal(df_numeric[selected_col].values, col_type)
            
            df_final[f"{selected_col}_CLEANED"] = cleaned_final
            
            st.dataframe(df_final.head(20))
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Hasil QC (.csv)", csv, f"qc_result_{selected_col}.csv", "text/csv")

    with tab3:
        st.header("⚙️ **Manual Cleaning Tools**")
        method_manual = st.selectbox("Metode Pembersihan Manual", ["Moving Average", "Hard Limit Filter", "Interpolation Only"])
        param = st.slider("Intensity / Window Size", 1, 21, 5, step=2)
        
        if st.button("🛠️ Apply Manual Tool"):
            raw_vals = df_numeric[selected_col].values
            col_type = processor.detected_columns.get(selected_col, "Data")
            
            # Tetap lakukan Hard QC dulu agar manual tool tidak error oleh 99.99
            base_data = processor.apply_hard_qc(raw_vals, col_type)
            
            if method_manual == "Moving Average":
                manual_res = processor.moving_average(base_data, window=param)
            elif method_manual == "Hard Limit Filter":
                manual_res = base_data # base_data sudah lewat hard filter
            else:
                manual_res = base_data
                
            fig_manual = go.Figure()
            fig_manual.add_trace(go.Scatter(y=raw_vals[:200], name="Raw", line=dict(color='red', dash='dot')))
            fig_manual.add_trace(go.Scatter(y=manual_res[:200], name="Manual Result", line=dict(color='blue')))
            st.plotly_chart(fig_manual)

if __name__ == "__main__":
    main()
