import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from fuzzywuzzy import fuzz
import warnings

warnings.filterwarnings('ignore')

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="OceanData QC Pro", page_icon="🌊", layout="wide")

st.markdown("""
# 🌊 **OceanData QC & Visualizer**
*Professional Quality Control untuk Data Salinitas, Suhu, dan Pasut*
""")

# 2. LOGIKA PEMROSESAN (THE "SENSE" MAKER)
class HydroProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.detected_cols = self._smart_detection()

    def _smart_detection(self):
        """Mengenali jenis kolom agar batas batas fisiknya masuk akal"""
        keywords = {
            'salinity': ['sal', 'psu', 'pss'],
            'temperature': ['temp', 'suhu', 't'],
            'water_level': ['wl', 'z', 'elev', 'pasut']
        }
        mapping = {}
        for col in self.df.columns:
            col_lower = str(col).lower()
            for dtype, keys in keywords.items():
                if any(fuzz.partial_ratio(col_lower, k) > 80 for k in keys):
                    mapping[col] = dtype
        return mapping

    def apply_despiking(self, data, col_type):
        """Hanya buang spike (99.99), pertahankan gerigi asli (Dinamika Laut)"""
        d = pd.Series(data).copy()
        
        # A. BATAS FISIK (Hard Limits)
        if col_type == 'salinity':
            d[(d < 2) | (d > 42)] = np.nan  # Salinitas laut normal 30-35
        elif col_type == 'temperature':
            d[(d < 5) | (d > 40)] = np.nan  # Suhu permukaan laut normal 25-32
        elif col_type == 'water_level':
            d[(d < -15) | (d > 15)] = np.nan
        
        # B. STATISTICAL DESPIKING (Z-Score Tinggi)
        # Pakai threshold 5 agar HANYA lonjakan gila yang kebuang.
        z = np.abs(stats.zscore(d, nan_policy='omit'))
        d[z > 5] = np.nan
        
        # C. INTERPOLASI LINEAR (Menambal yang bolong tanpa smoothing)
        return d.interpolate(method='linear').ffill().bfill().values

    def apply_smoothing(self, data, window):
        """Ini fitur opsional di tab Manual, bukan default"""
        return pd.Series(data).rolling(window=window, center=True).mean().ffill().bfill().values

# 3. INTERFACE UTAMA
def main():
    st.sidebar.header("📁 Step 1: Upload Data")
    file = st.sidebar.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx'])
    
    if not file:
        st.info("👈 Silakan upload data mentah kamu untuk memulai.")
        return

    # Load Data
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    df_num = df.select_dtypes(include=[np.number])
    proc = HydroProcessor(df_num)
    
    st.sidebar.success(f"Berhasil memuat {len(df)} baris.")
    sel_col = st.sidebar.selectbox("Pilih Parameter Utama", df_num.columns)
    col_type = proc.detected_cols.get(sel_col, 'generic')

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visualisasi & QC", "🧹 Auto-Clean & Export", "⚙️ Manual Tools"])

    with tab1:
        st.subheader("Analisis Kualitas Data")
        raw = df_num[sel_col].values
        clean = proc.apply_despiking(raw, col_type)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=("Perbandingan Raw vs Cleaned", "Residual (Error yang Dibuang)"))
        
        # Plot 1
        fig.add_trace(go.Scatter(y=raw, name="RAW (Original)", line=dict(color='red', width=1), opacity=0.4), row=1, col=1)
        fig.add_trace(go.Scatter(y=clean, name="CLEAN (No Spike)", line=dict(color='blue', width=1.2)), row=1, col=1)
        
        # Plot 2 (Residual)
        fig.add_trace(go.Scatter(y=raw-clean, name="Noise/Spike", line=dict(color='purple')), row=2, col=1)
        
        fig.update_layout(height=700, template="plotly_white", hovermode="x")
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning(f"**Logika:** Garis biru tetap 'bergerigi' karena itu adalah variasi alami lautmu. Hanya lonjakan vertikal (error) yang dibuang.")

    with tab2:
        st.subheader("Simpan Hasil Pembersihan")
        if st.button("Proses Seluruh Data QC"):
            df_qc = df.copy()
            for col in df_num.columns:
                ctype = proc.detected_cols.get(col, 'generic')
                df_qc[f"{col}_QC"] = proc.apply_despiking(df_num[col].values, ctype)
            
            st.write("Preview Data QC:")
            st.dataframe(df_qc.head())
            
            csv = df_qc.to_csv(index=False)
            st.download_button("💾 Download CSV Ter-QC", csv, "OceanData_Cleaned.csv", "text/csv")

    with tab3:
        st.subheader("Manual Processing (Gunakan dengan Bijak)")
        st.markdown("Jika kamu *memang* ingin menghaluskan data untuk keperluan tren jangka panjang:")
        
        window_size = st.slider("Window Size (Moving Average)", 1, 100, 1)
        
        raw_manual = df_num[sel_col].values
        # Pertama hapus spike dulu agar smoothing tidak rusak
        no_spike = proc.apply_despiking(raw_manual, col_type)
        # Baru apply smoothing
        smooth = proc.apply_smoothing(no_spike, window_size)
        
        fig_man = go.Figure()
        fig_man.add_trace(go.Scatter(y=no_spike, name="No Spike", line=dict(color='lightgrey')))
        fig_man.add_trace(go.Scatter(y=smooth, name=f"Smoothed (WA={window_size})", line=dict(color='orange', width=2)))
        st.plotly_chart(fig_man, use_container_width=True)

if __name__ == "__main__":
    main()
