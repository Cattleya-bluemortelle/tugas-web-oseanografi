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

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="HydroData Ultimate Cleaner",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
# 🌊 **HydroData Cleaner & Visualizer**
*Smart Quality Control untuk Data Oceanography (Anti-OverSmoothing)*
""")

# --- CLASS PROCESSOR (Lengkap dengan semua method kamu) ---
class HydroDataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.detected_columns = self._smart_column_detection()
    
    def _smart_column_detection(self):
        """Smart column detection dengan fuzzy matching seperti kodemu"""
        keywords = {
            'water_level': ['water_level', 'wl', 'elev', 'elevation', 'height', 'z', 'eta', 'pasut'],
            'salinity': ['sal', 'salinity', 'salin', 'pss', 'psu'],
            'temperature': ['temp', 'temperature', 'suhu', 't', 'degc'],
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
        """Deteksi outlier (Spike 99.99)"""
        if method == 'iqr':
            Q1 = np.nanpercentile(data, 25)
            Q3 = np.nanpercentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - 2.0 * IQR # Ditingkatkan ke 2.0 agar tidak terlalu sensitif
            upper = Q3 + 2.0 * IQR
            outliers = (data < lower) | (data > upper)
        elif method == 'zscore':
            # Pakai MAD (Median Absolute Deviation) karena lebih stabil untuk data sensor kotor
            median = np.nanmedian(data)
            ad = np.abs(data - median)
            mad = np.nanmedian(ad)
            outliers = ad > (5 * mad) # Threshold 5 agar gerigi asli aman
        return outliers
    
    def auto_noise_removal(self, data, col_type):
        """Sistem otomatis: HANYA ngilangin spike, TIDAK smoothing total"""
        outliers = self.detect_outliers(data, 'zscore')
        
        # Jika banyak outlier (tiang-tiang 99.99)
        if np.sum(outliers) > 0:
            cleaned = self.remove_outliers(data.copy())
            method = "Despiking (Robust Z-Score)"
        else:
            # Jika data 'clean' tapi berisik banget, baru smoothing tipis
            cleaned = self.moving_average(data, window=3)
            method = "Light Moving Average (Window=3)"
        
        return cleaned, method
    
    def remove_outliers(self, data):
        """Hapus spike raksasa dan tambal pakai interpolasi linear"""
        outliers = self.detect_outliers(data, 'zscore')
        data_clean = data.copy()
        data_clean[outliers] = np.nan
        # Interpolasi linear menjaga bentuk zigzag/gerigi asli
        return pd.Series(data_clean).interpolate(method='linear').ffill().bfill().values
    
    def moving_average(self, data, window=5):
        return pd.Series(data).rolling(window=window, center=True).mean().ffill().bfill().values
    
    def low_pass_filter(self, data, cutoff=0.1, fs=1.0):
        nyquist = fs / 2
        norm_cutoff = np.clip(cutoff / nyquist, 0.001, 0.999)
        b, a = signal.butter(4, norm_cutoff, btype='low')
        return signal.filtfilt(b, a, data)
    
    def band_pass_filter(self, data, lowcut=0.05, highcut=0.3, fs=1.0):
        nyquist = fs / 2
        b, a = signal.butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
        return signal.filtfilt(b, a, data)
    
    def high_pass_filter(self, data, cutoff=0.05, fs=1.0):
        nyquist = fs / 2
        b, a = signal.butter(4, cutoff/nyquist, btype='high')
        return signal.filtfilt(b, a, data)
    
    def interpolate_data(self, data):
        return pd.Series(data).interpolate(method='linear').ffill().bfill().values

    def curve_fit_sine(self, x, y):
        """Khusus untuk data pasang surut"""
        def sine_func(x, A, B, C, D):
            return A * np.sin(B * x + C) + D
        try:
            popt, _ = curve_fit(sine_func, x, y, p0=[np.std(y), 0.01, 0, np.mean(y)], maxfev=2000)
            return sine_func(x, *popt)
        except: return y

# --- FUNGSI VISUALISASI (Sesuai kodemu yang kompleks) ---
def create_visualization(df, processor, selected_col):
    col_data = df[selected_col].values
    time_idx = np.arange(len(col_data))
    col_type = processor.detected_columns.get(selected_col, 'generic')
    
    # Auto cleaning (Versi yang sudah diperbaiki logikanya)
    cleaned_data, method = processor.auto_noise_removal(col_data, col_type)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Raw vs Auto-QC (Despiking)', 'Methods Comparison (Zoom)', 
                       'Frequency Spectrum (FFT)', 'Residuals (Removed Spikes)')
    )
    
    # 1. Full Plot
    fig.add_trace(go.Scatter(y=col_data, name='Raw', line=dict(color='red', width=0.8), opacity=0.4), row=1, col=1)
    fig.add_trace(go.Scatter(y=cleaned_data, name='QC Clean', line=dict(color='blue', width=1.2)), row=1, col=1)
    
    # 2. Zoom Plot (300 points)
    z = min(300, len(col_data))
    fig.add_trace(go.Scatter(y=col_data[:z], name='Raw Zoom', line=dict(color='red', dash='dot')), row=1, col=2)
    fig.add_trace(go.Scatter(y=cleaned_data[:z], name='Clean Zoom', line=dict(color='blue')), row=1, col=2)
    
    # 3. Spectrum
    n = min(len(cleaned_data), 512)
    freq = fftfreq(n, 1)[:n//2]
    spectrum = np.abs(fft(cleaned_data[:n]))[:n//2]
    fig.add_trace(go.Scatter(x=freq, y=spectrum, name='Spectrum', fill='tozeroy'), row=2, col=1)
    
    # 4. Residuals
    fig.add_trace(go.Scatter(y=col_data - cleaned_data, name='Residuals', line=dict(color='purple')), row=2, col=2)
    
    fig.update_layout(height=800, title_text=f"📊 Analisis Detail: {selected_col} | Method: {method}", template="plotly_white")
    return fig, cleaned_data, method

# --- MAIN APP ---
def main():
    st.sidebar.header("📁 **Upload Data Source**")
    uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx'])
    
    if not uploaded_file:
        st.info("👋 Silakan upload file data oceanography kamu (CSV/Excel) di sidebar.")
        st.stop()
    
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df_numeric = df.select_dtypes(include=[np.number]).dropna(how='all')
        processor = HydroDataProcessor(df_numeric)
        st.sidebar.success(f"✅ Data loaded: {len(df_numeric)} baris")
    except Exception as e:
        st.error(f"❌ Error: {e}"); st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📈 **Visualisasi Pro**", "🧹 **Auto QC & Export**", "⚙️ **Manual Processing**"])
    
    with tab1:
        st.header("📈 **Analisis & Visualisasi**")
        selected_col = st.selectbox("Pilih Parameter", options=df_numeric.columns)
        if st.button("🎨 Generate Visualisasi Lengkap", type="primary"):
            fig, _, _ = create_visualization(df_numeric, processor, selected_col)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🧹 **Auto Quality Control**")
        selected_col_auto = st.selectbox("Parameter untuk Auto-QC", options=df_numeric.columns, key="auto")
        if st.button("🧠 Jalankan Auto-QC (Despiking)", type="primary"):
            raw = df_numeric[selected_col_auto].values
            clean, method = processor.auto_noise_removal(raw, processor.detected_columns.get(selected_col_auto))
            
            c1, c2 = st.columns(2)
            c1.metric("Metode Terpilih", method)
            c2.metric("Spike Terdeteksi", int(np.sum(raw != clean)))
            
            df_final = df.copy()
            df_final[f"{selected_col_auto}_CLEAN"] = clean
            st.dataframe(df_final.head(10))
            st.download_button("💾 Download Hasil (.csv)", df_final.to_csv(index=False), "Ocean_QC_Data.csv")

    with tab3:
        st.header("⚙️ **Manual Filtering Tools**")
        col_manual = st.selectbox("Pilih Kolom", df_numeric.columns, key="man")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            method_man = st.selectbox("Metode", ["Moving Average", "Low-pass Filter", "High-pass Filter", "Band-pass", "Sine Fit", "Interpolation"])
        with c2:
            param = st.slider("Parameter (Window/Cutoff x100)", 1, 100, 5)
        with c3:
            st.write(" ")
            run = st.button("🛠️ Apply Filter")

        if run:
            data = df_numeric[col_manual].values
            # Tetap despike dulu sebelum difilter agar tidak rusak
            data = processor.remove_outliers(data)
            
            if method_man == "Moving Average": res = processor.moving_average(data, window=param)
            elif method_man == "Low-pass Filter": res = processor.low_pass_filter(data, cutoff=param/100)
            elif method_man == "High-pass Filter": res = processor.high_pass_filter(data, cutoff=param/100)
            elif method_man == "Sine Fit": res = processor.curve_fit_sine(np.arange(len(data)), data)
            else: res = processor.interpolate_data(data)
            
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(y=data[:500], name="QC Data (No Spike)", line=dict(color='lightgrey')))
            fig_m.add_trace(go.Scatter(y=res[:500], name="Filtered Result", line=dict(color='orange', width=2)))
            st.plotly_chart(fig_m, use_container_width=True)

if __name__ == "__main__":
    main()
