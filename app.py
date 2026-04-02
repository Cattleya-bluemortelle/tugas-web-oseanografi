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

# Mengabaikan warning agar tampilan Streamlit tetap bersih
warnings.filterwarnings('ignore')

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="HydroData QC Ultimate",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("# 🌊 **HydroData Cleaner & Visualizer (Expert QC)**")
st.markdown("---")

# 2. LOGIKA PEMROSESAN DATA PROFESIONAL
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

    def apply_qc_limits(self, data, col_type):
        """Langkah krusial: Membuang angka error sensor (Outlier Removal)"""
        data_clean = data.copy()
        
        # Batas Logis Oseanografi (Hard Limits) - Sesuaikan jika perlu
        # Salinitas: 0-45 PSU, Suhu: 0-40 C, Water Level: -2 sampai 15 m
        limits = {
            'Salinitas': (0, 45),
            'Suhu': (0, 40),
            'Water Level': (-5, 20)
        }
        
        low, high = limits.get(col_type, (data.min(), data.max()))
        
        # 1. Hard Filter: Buang angka error ekstrem (seperti 99.99 atau 0 yang tidak mungkin)
        data_clean[(data_clean > high) | (data_clean < low)] = np.nan
        
        # 2. Statistical Filter: Buang spike tajam (IQR method)
        # Menggunakan 2 * IQR agar data dinamika laut yang asli tidak ikut terbuang
        q1 = np.nanpercentile(data_clean, 25)
        q3 = np.nanpercentile(data_clean, 75)
        iqr = q3 - q1
        data_clean[(data_clean < (q1 - 2*iqr)) | (data_clean > (q3 + 2*iqr))] = np.nan
        
        # 3. Interpolasi: Mengisi gap kosong secara linear agar tren data tetap terjaga
        series = pd.Series(data_clean)
        return series.interpolate(method='linear').bfill().ffill().values

    def adaptive_smoothing(self, data):
        """Hanya melakukan smoothing jika data berisik (noise > threshold)"""
        diff_std = np.nanstd(np.diff(data))
        
        # Jika variasi antar data sangat rapat/berisik, lakukan smoothing tipis
        if diff_std > 0.05:
            # Moving Average window 3 (sangat ringan agar tidak merusak nilai asli)
            cleaned = pd.Series(data).rolling(window=3, center=True).mean().bfill().ffill().values
            method = "QC Filtered + Light Smoothing"
        else:
            cleaned = data
            method = "QC Filtered (Original Value Preserved)"
            
        return cleaned, method

# 3. FUNGSI VISUALISASI DASHBOARD
def create_dashboard(df, selected_col, processor):
    col_type = processor.detected_columns.get(selected_col, "General Data")
    raw_data = df[selected_col].values
    
    # PROSES QC & CLEANING
    cleaned_data, method_name = processor.adaptive_smoothing(
        processor.apply_qc_limits(raw_data, col_type)
    )
    
    time_idx = np.arange(len(raw_data))
    
    # Layout Subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Full Time-Series: {selected_col}', 
            'Detailed Zoom (100 Data Points)', 
            'Frequency Spectrum (FFT)', 
            'Cleaning Residual (Noise Removed)'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Panel 1: Original vs Cleaned (Full)
    fig.add_trace(go.Scatter(x=time_idx, y=raw_data, name='Raw Data (Ada Error)', line=dict(color='red', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_idx, y=cleaned_data, name='Cleaned (Final QC)', line=dict(color='green', width=2)), row=1, col=1)

    # Panel 2: Zoom (Agar lekukan data terlihat jelas)
    limit_zoom = min(len(raw_data), 100)
    fig.add_trace(go.Scatter(x=time_idx[:limit_zoom], y=raw_data[:limit_zoom], name='Raw (Zoom)', line=dict(color='red', width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_idx[:limit_zoom], y=cleaned_data[:limit_zoom], name='Cleaned (Zoom)', line=dict(color='green', width=2)), row=1, col=2)

    # Panel 3: FFT (Melihat periodisitas, penting untuk Pasut)
    n = min(len(cleaned_data), 1024)
    yf = fft(cleaned_data[:n] - np.mean(cleaned_data[:n])) # detrending tipis untuk FFT
    xf = fftfreq(n, 1)[:n//2]
    fig.add_trace(go.Scatter(x=xf, y=np.abs(yf[:n//2]), name='Freq Domain', fill='tozeroy', line=dict(color='orange')), row=2, col=1)

    # Panel 4: Residuals (Noise yang dibuang)
    residual = raw_data - cleaned_data
    fig.add_trace(go.Scatter(x=time_idx, y=residual, name='Noise/Residual', line=dict(color='purple')), row=2, col=2)

    # Update Axes Labels
    fig.update_xaxes(title_text="Data Index", row=1, col=1)
    fig.update_yaxes(title_text=f"Value ({col_type})", row=1, col=1)
    fig.update_xaxes(title_text="Data Index (First 100)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Data Index", row=2, col=2)

    fig.update_layout(height=850, template="plotly_white", title=f"<b>QC Dashboard: {selected_col}</b> | Method: {method_name}", showlegend=True)
    return fig, cleaned_data

# 4. MAIN APP INTERFACE
def main():
    st.sidebar.header("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload CSV atau Excel", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            # Membaca file dengan format desimal titik (sesuai setting excel user)
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Filter hanya kolom angka saja
            df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
            
            if df_numeric.empty:
                st.error("Gagal mendeteksi kolom numerik. Pastikan format angka sudah benar (menggunakan titik sebagai desimal).")
                return

            processor = HydroDataProcessor(df_numeric)
            
            st.sidebar.success(f"✅ Berhasil memuat {len(df_numeric)} baris data.")
            
            # List kolom yang terdeteksi secara cerdas
            all_cols = df_numeric.columns.tolist()
            selected_col = st.sidebar.selectbox("Pilih Parameter Oseanografi:", all_cols)
            
            if selected_col:
                tab1, tab2 = st.tabs(["📈 Analisis QC & Visualisasi", "📥 Hasil Akhir & Download"])
                
                with tab1:
                    with st.spinner('Menjalankan prosedur QC...'):
                        fig, cleaned_result = create_dashboard(df_numeric, selected_col, processor)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Info Statistik
                        avg_val = np.mean(cleaned_result)
                        st.info(f"💡 **Hasil Analisis:** Rata-rata nilai {selected_col} setelah QC adalah **{avg_val:.2f}**. Nilai ini sudah dibersihkan dari spike error.")

                with tab2:
                    st.subheader("Ekspor Data Bersih")
                    df_final = df.copy()
                    df_final[f"{selected_col}_CLEANED_QC"] = cleaned_result
                    
                    st.markdown("Berikut adalah cuplikan data hasil pembersihan:")
                    st.dataframe(df_final.head(20))
                    
                    csv = df_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Data Hasil QC (.csv)",
                        data=csv,
                        file_name=f"hasil_qc_{selected_col}.csv",
                        mime='text/csv'
                    )
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
    else:
        st.info("Silakan unggah file CSV atau Excel kamu di sidebar untuk memulai proses Quality Control.")

if __name__ == "__main__":
    main()
