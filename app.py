import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from fuzzywuzzy import fuzz
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="HydroData Ultimate Cleaner",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
# 🌊 **HydroData Cleaner & Visualizer**
*Smart Noise Removal untuk Water Level, Salinitas, & Suhu*
""")

class HydroDataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.detected_columns = self._smart_column_detection()
    
    def _smart_column_detection(self):
        """Smart column detection dengan fuzzy matching"""
        keywords = {
            'water_level': ['water_level', 'wl', 'elev', 'elevation', 'height', 'z', 'pasang', 'surut'],
            'salinity': ['sal', 'salinity', 'salin', 'pss', 'psu'],
            'temperature': ['temp', 'temperature', 'suhu', 't'],
            'time': ['time', 'datetime', 'date', 'waktu']
        }
        
        mapping = {}
        for col in self.df.columns:
            col_lower = str(col).lower()
            best_score = 0
            best_type = None
            
            for data_type, words in keywords.items():
                for word in words:
                    score = fuzz.ratio(col_lower, word)
                    if score > best_score:
                        best_score = score
                        best_type = data_type
            
            if best_score > 25:
                mapping[col] = best_type
        
        return mapping
    
    def detect_outliers(self, data, method='iqr'):
        """Deteksi outlier dengan berbagai metode"""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = (data < lower) | (data > upper)
        elif method == 'zscore':
            z = np.abs(stats.zscore(data, nan_policy='omit'))
            outliers = z > 3
        return outliers
    
    def auto_noise_removal(self, data, time):
        """Sistem otomatis pilih metode terbaik"""
        outliers_iqr = np.sum(self.detect_outliers(data, 'iqr'))
        data_std = np.std(np.diff(data))
        
        if outliers_iqr > len(data) * 0.05:  # Jika banyak spike/outlier
            cleaned = self.remove_outliers(data.copy())
            method = "Outlier Removal (IQR)"
        elif data_std > np.std(data) * 0.3:  # Jika data sangat kasar/bergerigi
            cleaned = self.low_pass_filter(data, cutoff=0.1)
            method = "Low-pass Filter"
        else:  # Jika hanya noise halus
            cleaned = self.moving_average(data, window=5)
            method = "Moving Average"
        
        return cleaned, method
    
    def remove_outliers(self, data):
        """Hapus outlier dan tambal dengan interpolasi"""
        outliers = self.detect_outliers(data)
        data_clean = data.copy()
        data_clean[outliers] = np.nan
        return pd.Series(data_clean).interpolate(method='linear').ffill().bfill().values
    
    def moving_average(self, data, window=5):
        return pd.Series(data).rolling(window=window, center=True).mean().ffill().bfill().values
    
    def low_pass_filter(self, data, cutoff=0.1, fs=1.0):
        nyquist = fs / 2
        normal_cutoff = np.clip(cutoff / nyquist, 0.01, 0.99)
        b, a = signal.butter(4, normal_cutoff, btype='low')
        return signal.filtfilt(b, a, data)

    def interpolate_data(self, data):
        return pd.Series(data).interpolate(method='linear').ffill().bfill().values

def create_visualization(df, processor, selected_col):
    col_data = df[selected_col].values
    time_idx = np.arange(len(col_data))
    
    cleaned_data, method = processor.auto_noise_removal(col_data, time_idx)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data Original vs Auto-Cleaned', 'Comparison Methods (Zoom)', 
                       'Frequency Spectrum', 'Residuals (Noise Removed)')
    )
    
    # Plot 1: Full Comparison
    fig.add_trace(go.Scatter(y=col_data, name='Original', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(y=cleaned_data, name='Cleaned', line=dict(color='green', width=1.5)), row=1, col=1)
    
    # Plot 2: Zoom 200 points
    z = min(200, len(col_data))
    fig.add_trace(go.Scatter(y=col_data[:z], name='Original (Zoom)', line=dict(color='red', dash='dot')), row=1, col=2)
    fig.add_trace(go.Scatter(y=cleaned_data[:z], name='Cleaned (Zoom)', line=dict(color='green')), row=1, col=2)
    
    # Plot 3: Spectrum
    n = min(len(col_data), 512)
    freq = fftfreq(n, 1)[:n//2]
    spectrum = np.abs(fft(cleaned_data[:n]))[:n//2]
    fig.add_trace(go.Scatter(x=freq, y=spectrum, name='Spectrum', fill='tozeroy'), row=2, col=1)
    
    # Plot 4: Residuals
    fig.add_trace(go.Scatter(y=col_data - cleaned_data, name='Residuals', line=dict(color='purple')), row=2, col=2)
    
    fig.update_layout(height=800, template="plotly_white", title_text=f"Analisis Parameter: {selected_col}")
    return fig, cleaned_data, method

def main():
    st.sidebar.header("📁 **Upload Data**")
    uploaded_file = st.sidebar.file_uploader("Pilih CSV/XLSX", type=['csv', 'xlsx'])
    
    if not uploaded_file:
        st.info("👆 Upload file CSV atau Excel di samping untuk mulai memproses data.")
        st.stop()
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        df_numeric = df.select_dtypes(include=[np.number]).copy()
        if df_numeric.empty:
            st.error("❌ Tidak ditemukan data numerik di file kamu.")
            st.stop()
            
        processor = HydroDataProcessor(df_numeric)
        st.sidebar.success(f"✅ Berhasil memuat {len(df_numeric)} baris data.")
        
    except Exception as e:
        st.error(f"❌ Error saat membaca file: {e}")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📈 **Visualisasi**", "🧹 **Auto Clean**", "⚙️ **Manual Tools**"])
    
    with tab1:
        st.header("📈 **Visualisasi & Analisis**")
        selected_col = st.selectbox("Pilih Parameter", options=df_numeric.columns)
        if st.button("🎨 Generate Grafik", type="primary"):
            fig, _, method = create_visualization(df_numeric, processor, selected_col)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Metode otomatis yang disarankan: **{method}**")
            
    with tab2:
        st.header("🧹 **Auto Noise Removal**")
        selected_col_auto = st.selectbox("Pilih Parameter untuk Pembersihan", options=df_numeric.columns, key="auto")
        if st.button("🧠 Jalankan Pembersihan Otomatis", type="primary"):
            data = df_numeric[selected_col_auto].values
            cleaned, method = processor.auto_noise_removal(data, np.arange(len(data)))
            
            st.metric("Metode Digunakan", method)
            
            df_final = df.copy()
            df_final[f"{selected_col_auto}_CLEAN"] = cleaned
            
            st.dataframe(df_final.head(10))
            csv = df_final.to_csv(index=False)
            st.download_button("💾 Download Hasil (.csv)", csv, f"cleaned_{selected_col_auto}.csv", "text/csv")
            
    with tab3:
        st.header("⚙️ **Manual Cleaning Tools**")
        col_manual = st.selectbox("Pilih Parameter", df_numeric.columns, key="manual")
        
        c1, c2 = st.columns(2)
        with c1:
            method_man = st.selectbox("Metode", ["Moving Average", "Low-pass Filter", "Interpolation", "Outlier Removal"])
        with c2:
            param = st.slider("Parameter (Window/Intensity)", 1, 50, 5)
            
        if st.button("🛠️ Terapkan Manual"):
            raw = df_numeric[col_manual].values
            if method_man == "Moving Average":
                res = processor.moving_average(raw, window=param)
            elif method_man == "Low-pass Filter":
                res = processor.low_pass_filter(raw, cutoff=param/100)
            elif method_man == "Interpolation":
                res = processor.interpolate_data(raw)
            else:
                res = processor.remove_outliers(raw)
            
            fig_man = go.Figure()
            fig_man.add_trace(go.Scatter(y=raw[:500], name="Original", line=dict(color="red")))
            fig_man.add_trace(go.Scatter(y=res[:500], name="Hasil Manual", line=dict(color="blue")))
            st.plotly_chart(fig_man, use_container_width=True)

if __name__ == "__main__":
    main()
