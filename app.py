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

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(page_title="HydroData Ultimate Cleaner", page_icon="🌊", layout="wide")

st.markdown("# 🌊 **HydroData Cleaner & Visualizer (Complete Edition)**")
st.markdown("*Data aslimu tetap prioritas, hanya error yang dibersihkan.*")

class HydroDataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.detected_columns = self._smart_column_detection()
    
    def _smart_column_detection(self):
        """Mendeteksi tipe kolom agar filter tidak salah sasaran"""
        keywords = {
            'water_level': ['water_level', 'wl', 'elev', 'elevation', 'height', 'z', 'pasut'],
            'salinity': ['sal', 'salinity', 'salin', 'pss', 'psu'],
            'temperature': ['temp', 'temperature', 'suhu', 't'],
            'time': ['time', 'datetime', 'date', 'waktu']
        }
        mapping = {}
        for col in self.df.columns:
            col_lower = str(col).lower()
            for data_type, words in keywords.items():
                if any(fuzz.partial_ratio(col_lower, w) > 75 for w in words):
                    mapping[col] = data_type
                    break
        return mapping

    def apply_qc(self, data, col_type):
        """LOGIKA UTAMA: Hanya buang yang benar-benar error (Outlier)"""
        data_qc = data.copy()
        
        # 1. Hard Limits (Batas Fisik Laut)
        limits = {'salinity': (0, 45), 'temperature': (0, 40), 'water_level': (-10, 20)}
        low, high = limits.get(col_type, (np.nanmin(data), np.nanmax(data)))
        
        # Tandai sebagai NaN jika di luar batas (seperti 99.99)
        outliers = (data_qc > high) | (data_qc < low)
        
        # 2. Statistical Spike Removal (Z-Score)
        # Hanya buang yang jaraknya 3x standar deviasi (spike tajam)
        z_scores = np.abs(stats.zscore(data_qc, nan_policy='omit'))
        spikes = z_scores > 3
        
        data_qc[outliers | spikes] = np.nan
        
        # Jika tidak ada error, kembalikan data asli tanpa proses apapun
        if np.isnan(data_qc).sum() == 0:
            return data, "No Changes (Data Clean)"
            
        # Jika ada error, lakukan interpolasi linear untuk menambal lubang
        cleaned = pd.Series(data_qc).interpolate(method='linear').bfill().ffill().values
        return cleaned, "Outlier Removed & Interpolated"

    # --- Kumpulan Manual Tools ---
    def moving_average(self, data, window=5):
        return pd.Series(data).rolling(window=window, center=True).mean().bfill().ffill().values

    def low_pass_filter(self, data, cutoff=0.1):
        b, a = signal.butter(4, cutoff, btype='low')
        return signal.filtfilt(b, a, data)

    def interpolate_data(self, data):
        return pd.Series(data).interpolate(method='linear').bfill().ffill().values

# --- Visualisasi Dashboard ---
def create_dashboard(df, selected_col, processor):
    col_type = processor.detected_columns.get(selected_col, "generic")
    raw_data = df[selected_col].values
    
    # Auto Process
    cleaned_data, method = processor.apply_qc(raw_data, col_type)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Raw vs Cleaned (Full)', 'Comparison Zoom (150 Pts)', 'Frequency Spectrum', 'Residual (Error Removed)')
    )
    
    idx = np.arange(len(raw_data))
    # Plot 1: Full
    fig.add_trace(go.Scatter(x=idx, y=raw_data, name='Raw', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=cleaned_data, name='Cleaned', line=dict(color='green')), row=1, col=1)
    
    # Plot 2: Zoom
    z = min(len(raw_data), 150)
    fig.add_trace(go.Scatter(x=idx[:z], y=raw_data[:z], name='Raw Zoom', line=dict(color='red', dash='dot')), row=1, col=2)
    fig.add_trace(go.Scatter(x=idx[:z], y=cleaned_data[:z], name='Cleaned Zoom', line=dict(color='green')), row=1, col=2)
    
    # Plot 3: Spectrum
    yf = fft(cleaned_data - np.mean(cleaned_data))
    xf = fftfreq(len(cleaned_data), 1)[:len(cleaned_data)//2]
    fig.add_trace(go.Scatter(x=xf, y=np.abs(yf[:len(cleaned_data)//2]), name='Freq'), row=2, col=1)
    
    # Plot 4: Residual
    fig.add_trace(go.Scatter(x=idx, y=raw_data - cleaned_data, name='Noise'), row=2, col=2)
    
    fig.update_layout(height=800, template="plotly_white")
    return fig, cleaned_data, method

def main():
    st.sidebar.header("📁 Upload Data")
    file = st.sidebar.file_uploader("Pilih CSV/XLSX", type=['csv', 'xlsx'])
    
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df_num = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        processor = HydroDataProcessor(df_num)
        
        selected_col = st.sidebar.selectbox("Pilih Parameter", df_num.columns)
        
        tab1, tab2, tab3 = st.tabs(["📈 Visualisasi", "🧹 Auto Clean", "⚙️ Manual Tools"])
        
        with tab1:
            if selected_col:
                fig, cleaned, method = create_dashboard(df_num, selected_col, processor)
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Metode Terdeteksi: {method}")

        with tab2:
            st.header("Otomatis Bersihkan & Download")
            if st.button("Jalankan Auto-Clean Seluruh Data"):
                df_clean = df.copy()
                col_type = processor.detected_columns.get(selected_col, "generic")
                res, _ = processor.apply_qc(df_num[selected_col].values, col_type)
                df_clean[f"{selected_col}_CLEANED"] = res
                st.dataframe(df_clean.head())
                st.download_button("Download CSV", df_clean.to_csv(index=False), "data_bersih.csv")

        with tab3:
            st.header("Manual Cleaning (Gunakan jika Auto kurang pas)")
            m_method = st.selectbox("Pilih Tool", ["Moving Average", "Low-pass Filter", "Interpolasi Saja"])
            param = st.slider("Intensity", 1, 25, 5)
            
            if st.button("Apply Manual"):
                data = df_num[selected_col].values
                if m_method == "Moving Average": res = processor.moving_average(data, param)
                elif m_method == "Low-pass Filter": res = processor.low_pass_filter(data, param/100)
                else: res = processor.interpolate_data(data)
                
                f_manual = go.Figure()
                f_manual.add_trace(go.Scatter(y=data[:300], name="Original"))
                f_manual.add_trace(go.Scatter(y=res[:300], name="Manual Result"))
                st.plotly_chart(f_manual)

if __name__ == "__main__":
    main()
