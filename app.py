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

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="HydroData QC Pro", page_icon="🌊", layout="wide")

st.markdown("# 🌊 **HydroData Cleaner & Visualizer (Final QC)**")
st.markdown("---")

class HydroDataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.detected_columns = self._smart_column_detection()
    
    def _smart_column_detection(self):
        keywords = {
            'Salinitas': ['sal', 'salinity', 'psu', 'pss'],
            'Suhu': ['temp', 'temperature', 'suhu', 't', 'degc'],
            'Water Level': ['water_level', 'wl', 'elev', 'elevation', 'height', 'z', 'pasut'],
            'Waktu': ['time', 'datetime', 'date', 'waktu']
        }
        mapping = {}
        for col in self.df.columns:
            col_lower = str(col).lower()
            for data_type, words in keywords.items():
                if any(fuzz.ratio(col_lower, w) > 65 or w in col_lower for w in words):
                    mapping[col] = data_type
                    break
        return mapping

    def remove_outliers_professionally(self, data, col_type):
        """Langkah QC: Membuang angka error sensor (seperti 99.99)"""
        data_clean = data.copy()
        
        # Batas Logis Oseanografi (Hard Limits)
        limits = {
            'Salinitas': (0, 45),
            'Suhu': (0, 40),
            'Water Level': (-2, 10)
        }
        
        low, high = limits.get(col_type, (data.min(), data.max()))
        
        # 1. Buang data yang secara fisik mustahil (e.g. 99.99 psu)
        data_clean[(data_clean > high) | (data_clean < low)] = np.nan
        
        # 2. Buang Spike Statistik (IQR)
        q1 = np.nanpercentile(data_clean, 25)
        q3 = np.nanpercentile(data_clean, 75)
        iqr = q3 - q1
        data_clean[(data_clean < (q1 - 2*iqr)) | (data_clean > (q3 + 2*iqr))] = np.nan
        
        # 3. Interpolasi (Mengisi lubang dengan tren linear, bukan rata-rata)
        return pd.Series(data_clean).interpolate(method='linear').bfill().ffill().values

    def moving_average(self, data, window=5):
        return pd.Series(data).rolling(window=window, center=True).mean().bfill().ffill().values

    def auto_clean(self, data, col_type):
        # LANGKAH 1: WAJIB QC Outlier dulu (Agar 99.99 tidak merusak rata-rata)
        qc_data = self.remove_outliers_professionally(data, col_type)
        
        # LANGKAH 2: Cek noise (getaran kecil)
        noise_level = np.nanstd(np.diff(qc_data))
        
        # LANGKAH 3: Hanya smoothing jika data sangat 'berisik'
        if noise_level > 0.1:
            cleaned = self.moving_average(qc_data, window=3)
            method = "QC Passed + Light Smoothing"
        else:
            cleaned = qc_data
            method = "QC Passed (Original Preserved)"
            
        return cleaned, method

# 3. VISUALISASI
def create_dashboard(df, selected_col, processor):
    col_type = processor.detected_columns.get(selected_col, "Data")
    raw_data = df[selected_col].values
    
    # Jalankan proses pembersihan
    cleaned_data, method_name = processor.auto_clean(raw_data, col_type)
    time_idx = np.arange(len(raw_data))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Perbandingan Data Asli vs QC', 'Zoom Analisis (100 Data Pertama)', 
                       'Spektrum Frekuensi (FFT)', 'Noise/Residual yang Dibuang'),
        vertical_spacing=0.12
    )

    # Plot 1: Full Data
    fig.add_trace(go.Scatter(x=time_idx, y=raw_data, name='Raw (Ada Error)', line=dict(color='red', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_idx, y=cleaned_data, name='Cleaned (QC)', line=dict(color='green', width=2)), row=1, col=1)

    # Plot 2: Zoom (Agar terlihat lekukannya)
    fig.add_trace(go.Scatter(x=time_idx[:100], y=raw_data[:100], name='Raw Zoom', line=dict(color='red', width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_idx[:100], y=cleaned_data[:100], name='Cleaned Zoom', line=dict(color='green', width=2)), row=1, col=2)

    # Plot 3: FFT
    n = min(len(cleaned_data), 512)
    yf = fft(cleaned_data[:n])
    xf = fftfreq(n, 1)[:n//2]
    fig.add_trace(go.Scatter(x=xf, y=np.abs(yf[:n//2]), name='Frequency', fill='tozeroy'), row=2, col=1)

    # Plot 4: Residual
    residual = raw_data - cleaned_data
    fig.add_trace(go.Scatter(x=time_idx, y=residual, name='Residual/Noise', line=dict(color='purple')), row=2, col=2)

    # LABELING SUMBU (Agar Tidak Bingung)
    fig.update_xaxes(title_text="Indeks Waktu", row=1, col=1)
    fig.update_yaxes(title_text=f"{col_type} ({selected_col})", row=1, col=1)
    fig.update_xaxes(title_text="Indeks Waktu", row=1, col=2)
    fig.update_yaxes(title_text="Nilai", row=1, col=2)
    fig.update_xaxes(title_text="Frekuensi", row=2, col=1)
    fig.update_xaxes(title_text="Indeks Waktu", row=2, col=2)

    fig.update_layout(height=800, template="plotly_white", title=f"Analisis Oseanografi: {selected_col} | Metode: {method_name}")
    return fig, cleaned_data

def main():
    st.sidebar.header("📁 File Input")
    file = st.sidebar.file_uploader("Upload Data Oseanografi", type=['csv', 'xlsx'])

    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df_num = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        
        processor = HydroDataProcessor(df_num)
        selected_col = st.sidebar.selectbox("Pilih Kolom:", df_num.columns)
        
        if selected_col:
            tab1, tab2 = st.tabs(["📈 Analisis QC", "📥 Download"])
            
            with tab1:
                fig, cleaned_res = create_dashboard(df_num, selected_col, processor)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cek apakah hasil masuk akal
                avg_val = np.mean(cleaned_res)
                st.info(f"💡 Rata-rata {selected_col} setelah QC: **{avg_val:.2f}**. Jika angka ini sesuai dengan kondisi lapangan (30-an psu), maka data aman.")

            with tab2:
                df_final = df.copy()
                df_final[f"{selected_col}_QC"] = cleaned_res
                st.download_button("💾 Download Hasil Bersih (.csv)", df_final.to_csv(index=False), "data_qc_hasil.csv", "text/csv")
                st.dataframe(df_final.head(20))
    else:
        st.info("Silakan upload file di sidebar untuk memulai.")

if __name__ == "__main__":
    main()
