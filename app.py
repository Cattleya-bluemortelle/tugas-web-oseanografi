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
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="HydroData Cleaner",
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
        self.original_df = df.copy()
        self.detected_columns = self._smart_column_detection()
    
    def _smart_column_detection(self):
        """Smart column detection dengan fuzzy matching"""
        keywords = {
            'water_level': ['water_level', 'wl', 'elev', 'elevation', 'height', 'z', 'eta'],
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
            z = np.abs(stats.zscore(data))
            outliers = z > 3
        return outliers
    
    def auto_noise_removal(self, data, time):
        """Sistem otomatis pilih metode terbaik"""
        outliers_iqr = np.sum(self.detect_outliers(data, 'iqr'))
        data_std = np.std(np.diff(data))
        
        if outliers_iqr > len(data) * 0.1:  # Banyak outlier
            cleaned = self.remove_outliers(data.copy())
            method = "Outlier Removal (IQR)"
        elif data_std > np.std(data) * 0.5:  # High frequency noise
            cleaned = self.low_pass_filter(data, fs=1.0)
            method = "Low-pass Filter"
        else:  # Mild noise
            cleaned = self.moving_average(data, window=5)
            method = "Moving Average"
        
        return cleaned, method
    
    def remove_outliers(self, data):
        """Hapus outlier dan interpolasi"""
        outliers = self.detect_outliers(data)
        data_clean = data.copy()
        data_clean[outliers] = np.nan
        # FIXED: Menggunakan .bfill().ffill()
        return pd.Series(data_clean).interpolate(method='linear').bfill().ffill().values
    
    def moving_average(self, data, window=5):
        """Moving average smoothing (FIXED Pandas 2.0+)"""
        # FIXED: Menggunakan .bfill().ffill()
        return pd.Series(data).rolling(window=window, center=True).mean().bfill().ffill().values
    
    def low_pass_filter(self, data, cutoff=0.1, fs=1.0):
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low')
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
        """Interpolasi linear (FIXED Pandas 2.0+)"""
        # FIXED: Menggunakan .bfill().ffill()
        return pd.Series(data).interpolate(method='linear').bfill().ffill().values

def create_visualization(df, processor, selected_col):
    col_data = df[selected_col].dropna()
    time_idx = np.arange(len(col_data))
    cleaned_data, method = processor.auto_noise_removal(col_data.values, time_idx)
    
    methods = {
        'Original': col_data.values,
        f'Auto-Clean ({method})': cleaned_data,
        'Moving Avg (5)': processor.moving_average(col_data.values),
        'Low-pass': processor.low_pass_filter(col_data.values),
        'Interpolation': processor.interpolate_data(col_data.values)
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Original vs Auto-Cleaned', 'Comparison Methods', 
                       'Frequency Spectrum', 'Residual Analysis')
    )
    
    # Plot 1
    fig.add_trace(go.Scatter(x=time_idx[:500], y=col_data.values[:500], name='Original', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_idx[:500], y=cleaned_data[:500], name='Auto-Cleaned', line=dict(color='green')), row=1, col=1)
    
    # Plot 2
    colors = px.colors.qualitative.Set1
    for i, (name, data) in enumerate(methods.items()):
        fig.add_trace(go.Scatter(x=time_idx[:300], y=data[:300], name=name, line=dict(color=colors[i % len(colors)])), row=1, col=2)
    
    # Plot 3
    freq = fftfreq(256, 1)[:128]
    spectrum_orig = np.abs(fft(col_data.values[:256]))[:128]
    spectrum_clean = np.abs(fft(cleaned_data[:256]))[:128]
    fig.add_trace(go.Scatter(x=freq, y=spectrum_orig, name='Orig Spectrum'), row=2, col=1)
    fig.add_trace(go.Scatter(x=freq, y=spectrum_clean, name='Clean Spectrum'), row=2, col=1)
    
    # Plot 4
    residuals = col_data.values[:300] - cleaned_data[:300]
    fig.add_trace(go.Scatter(x=time_idx[:300], y=residuals, name='Residuals', line=dict(color='purple')), row=2, col=2)

    # TAMBAHAN: LABEL SUMBU AGAR TIDAK BINGUNG
    fig.update_xaxes(title_text="Indeks Data", row=1, col=1)
    fig.update_yaxes(title_text=f"Nilai {selected_col}", row=1, col=1)
    fig.update_xaxes(title_text="Frekuensi", row=2, col=1)
    fig.update_yaxes(title_text="Amplitudo", row=2, col=1)

    fig.update_layout(height=800, title_text=f"📊 Analisis Noise: {selected_col}")
    return fig, cleaned_data, method

def main():
    st.sidebar.header("📁 **Upload Data**")
    uploaded_file = st.sidebar.file_uploader("Pilih CSV/XLSX", type=['csv', 'xlsx'])
    
    if not uploaded_file:
        st.info("👆 Upload file untuk memulai")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df_numeric = df.select_dtypes(include=[np.number]).dropna()
        processor = HydroDataProcessor(df_numeric)
        st.sidebar.success(f"✅ Loaded: {len(df_numeric)} baris")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📈 **Visualisasi**", "🧹 **Auto Clean**", "⚙️ **Manual Tools**"])
    
    with tab1:
        st.header("📈 Visualisasi Data")
        selected_col = st.selectbox("Pilih Parameter", options=list(processor.detected_columns.keys()))
        if st.button("🎨 Generate Visualisasi", type="primary"):
            fig, _, _ = create_visualization(df_numeric, processor, selected_col)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🧹 Auto Noise Removal")
        selected_col_auto = st.selectbox("Parameter Auto-Clean", options=list(processor.detected_columns.keys()), key="auto")
        if st.button("🧠 Auto Clean & Download", type="primary"):
            col_data = df_numeric[selected_col_auto].dropna()
            cleaned_data, method = processor.auto_noise_removal(col_data.values, np.arange(len(col_data)))
            
            st.metric("Metode Digunakan", method)
            df_clean = df_numeric.copy()
            df_clean[selected_col_auto] = cleaned_data
            st.download_button("💾 Download Hasil", df_clean.to_csv(index=False), f"cleaned_{selected_col_auto}.csv")
            
    with tab3:
        st.header("⚙️ Manual Cleaning Tools")
        col_manual = st.selectbox("Pilih Kolom", df_numeric.columns, key="manual")
        method_m = st.selectbox("Metode", ["Moving Average", "Low-pass Filter", "Interpolation", "Outlier Removal"])
        param = st.slider("Parameter", 1, 50, 5)
        
        if st.button("🛠️ Apply Manual Clean"):
            data = df_numeric[col_manual].values
            if method_m == "Moving Average": res = processor.moving_average(data, window=param)
            elif method_m == "Low-pass Filter": res = processor.low_pass_filter(data, cutoff=param/100)
            elif method_m == "Interpolation": res = processor.interpolate_data(data)
            else: res = processor.remove_outliers(data)
            
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(y=data[:1000], name='Original', line=dict(color='gray')))
            fig_m.add_trace(go.Scatter(y=res[:1000], name='Cleaned', line=dict(color='cyan')))
            st.plotly_chart(fig_m)

if __name__ == "__main__":
    main()
