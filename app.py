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

# Konfigurasi halaman
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
            'salinity': ['sal', 'salinity', 'salin', 'pss'],
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
        """Sistem otomatis pilih metode terbaik untuk noise removal"""
        outliers_iqr = np.sum(self.detect_outliers(data, 'iqr'))
        data_std = np.std(np.diff(data))
        
        if outliers_iqr > len(data) * 0.1:  
            cleaned = self.remove_outliers(data.copy())
            method = "Outlier Removal (IQR)"
        elif data_std > np.std(data) * 0.5:  
            cleaned = self.low_pass_filter(data, fs=1.0)
            method = "Low-pass Filter"
        else:  
            cleaned = self.moving_average(data, window=5)
            method = "Moving Average"
        
        return cleaned, method
    
    def remove_outliers(self, data):
        """Hapus outlier dan interpolasi"""
        outliers = self.detect_outliers(data)
        data_clean = data.copy()
        data_clean[outliers] = np.nan
        # Perbaikan fillna di sini
        return pd.Series(data_clean).interpolate(method='linear').bfill().ffill().values
    
    def moving_average(self, data, window=5):
        """Moving average smoothing - FIXED Pandas 2.0+ compatible"""
        # Perbaikan fillna di sini
        return pd.Series(data).rolling(window=window, center=True).mean().bfill().ffill().values
    
    def low_pass_filter(self, data, cutoff=0.1, fs=1.0):
        """Low-pass Butterworth filter"""
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low')
        return signal.filtfilt(b, a, data)
    
    def band_pass_filter(self, data, lowcut=0.05, highcut=0.3, fs=1.0):
        """Band-pass filter"""
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def high_pass_filter(self, data, cutoff=0.05, fs=1.0):
        """High-pass filter"""
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high')
        return signal.filtfilt(b, a, data)
    
    def interpolate_data(self, data):
        """Interpolasi linear - FIXED Pandas 2.0+ compatible"""
        # Perbaikan fillna di sini
        return pd.Series(data).interpolate(method='linear').bfill().ffill().values
    
    def curve_fit_sine(self, x, y):
        """Curve fitting dengan fungsi sine"""
        def sine_func(x, A, B, C, D):
            return A * np.sin(B * x + C) + D
        
        try:
            popt, _ = curve_fit(sine_func, x, y, maxfev=5000)
            return sine_func(x, *popt)
        except:
            return y

def create_visualization(df, processor, selected_col):
    """Buat visualisasi lengkap"""
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
        subplot_titles=('Data Original vs Auto-Cleaned', 'Comparison Methods', 
                       'Frequency Spectrum', 'Residual Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Original vs Auto-cleaned
    fig.add_trace(
        go.Scatter(x=time_idx[:500], y=col_data.values[:500], 
                  name='Original', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_idx[:500], y=cleaned_data[:500], 
                  name=f'Auto-Cleaned', line=dict(color='green')),
        row=1, col=1
    )
    
    # Plot 2: Comparison
    colors = px.colors.qualitative.Set1
    for i, (name, data) in enumerate(methods.items()):
        fig.add_trace(
            go.Scatter(x=time_idx[:300], y=data[:300], 
                      name=name, line=dict(color=colors[i % len(colors)])),
            row=1, col=2
        )
    
    # Plot 3: Spectrum
    freq = fftfreq(256, 1)[:128]
    spectrum_orig = np.abs(fft(col_data.values[:256]))[:128]
    spectrum_clean = np.abs(fft(cleaned_data[:256]))[:128]
    
    fig.add_trace(
        go.Scatter(x=freq, y=spectrum_orig, name='Original Spectrum'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=freq, y=spectrum_clean, name='Cleaned Spectrum'),
        row=2, col=1
    )
    
    # Plot 4: Residuals
    residuals = col_data.values[:300] - cleaned_data[:300]
    fig.add_trace(
        go.Scatter(x=time_idx[:300], y=residuals, 
                  name='Residuals', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=True, 
                      title_text=f"📊 {selected_col} - Noise Analysis")
    return fig, cleaned_data, method

def main():
    st.sidebar.header("📁 **Upload Data**")
    
    uploaded_file = st.sidebar.file_uploader(
        "Pilih CSV/XLSX", type=['csv', 'xlsx'],
        help="Upload data water level, salinitas, atau suhu"
    )
    
    if not uploaded_file:
        st.info("👆 Upload file CSV/XLSX untuk memulai")
        st.stop()
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Cari baris data numerik
        for i in range(min(10, len(df))):
            if df.iloc[i].apply(lambda x: pd.api.types.is_numeric_dtype(type(x)) or pd.isna(x)).all():
                df = df.iloc[i:].reset_index(drop=True)
                break
        
        df_numeric = df.select_dtypes(include=[np.number]).dropna()
        processor = HydroDataProcessor(df_numeric)
        
        st.sidebar.success(f"✅ Data loaded: {len(df_numeric)} rows")
        
    except Exception as e:
        st.error(f"❌ Error loading: {str(e)}")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📈 **Visualisasi**", "🧹 **Auto Clean**", "⚙️ **Manual Tools**"])
    
    with tab1:
        st.header("📈 **Visualisasi Data**")
        col_options = processor.detected_columns
        if not col_options:
            st.warning("Tidak ada kolom numerik yang terdeteksi secara otomatis.")
            selected_col = st.selectbox("Pilih Parameter Secara Manual", df_numeric.columns)
        else:
            selected_col = st.selectbox(
                "Pilih Parameter",
                options=list(col_options.keys()),
                format_func=lambda x: f"{x} ({col_options[x]})"
            )
        
        if st.button("🎨 **Generate Visualisasi**", type="primary"):
            fig, _, _ = create_visualization(df_numeric, processor, selected_col)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🧹 **Auto Noise Removal**")
        selected_col_auto = st.selectbox("Parameter untuk Auto-Clean", df_numeric.columns)
        
        if st.button("🧠 **Auto Clean & Download**", type="primary"):
            col_data = df_numeric[selected_col_auto].dropna()
            cleaned_data, method = processor.auto_noise_removal(col_data.values, np.arange(len(col_data)))
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Noise Reduction", f"{100*(1-np.std(cleaned_data)/np.std(col_data.values)):.1f}%")
            c2.metric("Method Used", method)
            c3.metric("Data Points", f"{len(cleaned_data):,}")
            
            # Download
            df_clean = df_numeric.copy()
            df_clean[selected_col_auto] = cleaned_data
            csv = df_clean.to_csv(index=False)
            st.download_button("💾 Download Cleaned Data", csv, f"cleaned_{selected_col_auto}.csv", "text/csv")
    
    with tab3:
        st.header("⚙️ **Manual Cleaning Tools**")
        col_manual = st.selectbox("Pilih Kolom", df_numeric.columns, key="manual_sel")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            method_manual = st.selectbox("Metode", ["Moving Average", "Low-pass Filter", "Interpolation", "Outlier Removal"])
        with m_col2:
            param = st.slider("Intensity Parameter", 1, 50, 5)
        
        if st.button("🛠️ **Apply Manual Clean**"):
            data = df_numeric[col_manual].values
            if method_manual == "Moving Average":
                res = processor.moving_average(data, window=param)
            elif method_manual == "Low-pass Filter":
                res = processor.low_pass_filter(data, cutoff=param/100)
            elif method_manual == "Interpolation":
                res = processor.interpolate_data(data)
            else:
                res = processor.remove_outliers(data)
            
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(y=data[:1000], name='Original', line=dict(color='gray')))
            fig_m.add_trace(go.Scatter(y=res[:1000], name='Cleaned', line=dict(color='cyan')))
            st.plotly_chart(fig_m)

if __name__ == "__main__":
    main()
