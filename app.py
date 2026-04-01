import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from scipy import signal
from scipy.fft import fft, fftfreq
from fuzzywuzzy import fuzz
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Ocean Dynamics Hub",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 **Ocean Dynamics Hub**")
st.markdown("---")

# Fungsi utilitas
@st.cache_data
def detect_numeric_row(df):
    """Mendeteksi baris pertama yang mengandung data numerik"""
    for i in range(min(10, len(df))):
        numeric_row = pd.to_numeric(df.iloc[i], errors='coerce').notna().all()
        if numeric_row:
            return i
    return 0

def smart_column_mapping(df, data_type):
    """Smart column mapping dengan fuzzy matching"""
    col_mapping = {}
    
    if data_type == "wave":
        keywords = ['elev', 'z', 'water_level', 'height', 'depth', 'pres', 'eta']
    else:  # current
        keywords = ['u', 'v', 'vel', 'arus', 'speed', 'dir', 'east', 'north']
    
    for col in df.columns:
        col_lower = str(col).lower()
        best_match = 0
        best_keyword = None
        
        for keyword in keywords:
            score = fuzz.ratio(col_lower, keyword)
            if score > best_match:
                best_match = score
                best_keyword = keyword
        
        if best_match > 30:  # Threshold fuzzy matching
            col_mapping[col] = best_keyword
    
    return col_mapping

def clean_data(df):
    """Membersihkan data: konversi teks rusak ke NaN dan hapus"""
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()
    return df_clean

def low_pass_filter(data, cutoff_freq=0.5, fs=1.0):
    """Low-pass filter untuk memisahkan sinyal gelombang murni"""
    nyquist = fs / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, data)

# Fungsi analisis gelombang
def analyze_waves(df, time_col, elev_col, depth):
    """Analisis lengkap gelombang"""
    df = df[[time_col, elev_col]].dropna()
    time = df[time_col].values
    elev = df[elev_col].values
    
    # Low-pass filter
    elev_filtered = low_pass_filter(elev, cutoff_freq=0.3, fs=1.0)
    
    # Hitung parameter gelombang
    Hs = 4 * np.std(elev_filtered)
    Hrms = np.sqrt(np.mean(elev_filtered**2))
    
    # Energi gelombang (ρ = 1025 kg/m³, g = 9.81 m/s²)
    rho = 1025
    g = 9.81
    E = (1/8) * rho * g * Hs**2
    
    # Klasifikasi perairan
    L = 1.56 * depth  # Aproksimasi panjang gelombang
    d_over_L = depth / L
    
    if d_over_L > 0.5:
        water_class = "Perairan Dalam"
    elif d_over_L > 0.05:
        water_class = "Perairan Transisi"
    else:
        water_class = "Perairan Dangkal"
    
    # Spektrum energi
    N = len(elev_filtered)
    T = time[-1] - time[0]
    freq = fftfreq(N, T/N)[:N//2]
    spectrum = np.abs(fft(elev_filtered))[:N//2]**2
    
    return {
        'Hs': Hs, 'Hrms': Hrms, 'E': E,
        'water_class': water_class, 'd_over_L': d_over_L,
        'time': time, 'elev_raw': elev, 'elev_filtered': elev_filtered,
        'freq': freq, 'spectrum': spectrum
    }

# Fungsi analisis arus
def analyze_currents(df, u_col=None, v_col=None):
    """Analisis lengkap arus"""
    df = df.dropna()
    
    # Auto unit conversion
    u_data = df[u_col].mean()
    v_data = df[v_col].mean()
    
    if u_data > 10 or v_data > 10:  # cm/s to m/s
        df[u_col] *= 0.01
        df[v_col] *= 0.01
        unit = "m/s (converted from cm/s)"
    else:
        unit = "m/s"
    
    # Statistik
    speed = np.sqrt(df[u_col]**2 + df[v_col]**2)
    avg_speed = speed.mean()
    max_speed = speed.max()
    
    return {
        'df': df, 'speed': speed.values, 'unit': unit,
        'avg_speed': avg_speed, 'max_speed': max_speed
    }

# Main App
def main():
    # Sidebar untuk input
    with st.sidebar:
        st.header("📁 **Input Data**")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV atau XLSX", 
            type=['csv', 'xlsx'],
            help="Upload file data gelombang atau arus"
        )
        
        st.markdown("---")
        st.header("⚙️ **Parameter**")
        
        depth = st.number_input("Kedalaman Laut (m)", min_value=0.1, value=20.0)
        
        if uploaded_file:
            # Baca file dengan header-agnostic
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Deteksi baris data numerik
                start_row = detect_numeric_row(df)
                if start_row > 0:
                    df = df.iloc[start_row:].reset_index(drop=True)
                
                df = clean_data(df)
                st.success(f"✅ Data berhasil dimuat: {len(df)} baris")
                st.dataframe(df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
                st.stop()
        else:
            st.info("👆 Silakan upload file terlebih dahulu")
            st.stop()
    
    # Tabs utama
    tab1, tab2 = st.tabs(["🌊 **Analisis Gelombang**", "💨 **Analisis Arus**"])
    
    # Auto-detect columns
    wave_cols = smart_column_mapping(df, "wave")
    current_cols = smart_column_mapping(df, "current")
    
    with tab1:
        st.header("🌊 **Analisis Gelombang**")
        
        col1, col2 = st.columns(2)
        with col1:
            time_col = st.selectbox(
                "Kolom Waktu", 
                options=df.columns.tolist(),
                index=0
            )
        with col2:
            elev_col = st.selectbox(
                "Kolom Elevasi", 
                options=[col for col in df.columns if col in wave_cols],
                index=0
            )
        
        if st.button("🚀 **Jalankan Analisis Gelombang**", type="primary"):
            with st.spinner("Menganalisis gelombang..."):
                results = analyze_waves(df, time_col, elev_col, depth)
            
            # Results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("**Tinggi Gelombang Signifikan (Hs)**", f"{results['Hs']:.2f} m")
            with col2:
                st.metric("**Hrms**", f"{results['Hrms']:.2f} m")
            with col3:
                st.metric("**Rapat Energi (E)**", f"{results['E']/1000:.1f} kJ/m²")
            
            st.markdown("**Klasifikasi Perairan:**")
            st.info(f"**{results['water_class']}** (d/L = {results['d_over_L']:.3f})")
            
            # Visualisasi
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sinyal Gelombang', 'Spektrum Energi', 
                              'Filtered vs Raw', 'Histogram Elevasi'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Time series
            fig.add_trace(
                go.Scatter(x=results['time'], y=results['elev_raw'], 
                         name='Raw', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=results['time'], y=results['elev_filtered'], 
                         name='Filtered', line=dict(color='red')),
                row=1, col=1
            )
            
            # Spektrum
            fig.add_trace(
                go.Scatter(x=results['freq'], y=results['spectrum'], 
                         name='Spektrum', line=dict(color='green')),
                row=1, col=2
            )
            
            # Filtered vs Raw comparison
            fig.add_trace(
                go.Scatter(x=results['elev_raw'], y=results['elev_filtered'], 
                         mode='markers', name='Data Points'),
                row=2, col=1
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=results['elev_filtered'], name='Elevasi Filtered',
                           nbinsx=30),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True, title_text="Analisis Gelombang Lengkap")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("💨 **Analisis Arus**")
        
        col1, col2 = st.columns(2)
        with col1:
            u_col = st.selectbox("Komponen U (Timur)", options=df.columns.tolist())
        with col2:
            v_col = st.selectbox("Komponen V (Utara)", options=df.columns.tolist())
        
        if st.button("🚀 **Jalankan Analisis Arus**", type="primary"):
            with st.spinner("Menganalisis arus..."):
                results = analyze_currents(df, u_col, v_col)
            
            # Statistik
            col1, col2 = st.columns(2)
            with col1:
                st.metric("**Kecepatan Rata-rata**", f"{results['avg_speed']:.3f} {results['unit']}")
                st.metric("**Kecepatan Maksimum**", f"{results['max_speed']:.3f} {results['unit']}")
            with col2:
                st.metric("**Standar Deviasi**", f"{results['speed'].std():.3f} {results['unit']}")
            
            # Visualisasi
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Vektor Arus (Quiver)', 'Histogram Kecepatan', 
                              'Time Series U', 'Time Series V')
            )
            
            # Quiver plot
            x_grid, y_grid = np.meshgrid(np.arange(0, 10, 1), np.arange(0, 10, 1))
            u_sample = results['df'][u_col].values[:100]
            v_sample = results['df'][v_col].values[:100]
            
            fig.add_trace(
                go.Scatter(x=np.arange(len(u_sample)), y=u_sample,
                          mode='markers', marker=dict(size=8, color=v_sample),
                          name='Vektor Arus'),
                row=1, col=1
            )
            
            # Histogram kecepatan
            fig.add_trace(
                go.Histogram(x=results['speed'], name='Kecepatan Arus',
                           nbinsx=30),
                row=1, col=2
            )
            
            # Time series U dan V
            time_idx = np.arange(len(results['df']))
            fig.add_trace(
                go.Scatter(x=time_idx, y=results['df'][u_col], name='U (Timur)'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_idx, y=results['df'][v_col], name='V (Utara)'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True, title_text="Analisis Arus Lengkap")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
