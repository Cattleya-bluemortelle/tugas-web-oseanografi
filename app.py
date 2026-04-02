import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from fuzzywuzzy import fuzz
import warnings

warnings.filterwarnings('ignore')

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="HydroData Raw vs QC", page_icon="🌊", layout="wide")

st.markdown("# 🌊 **HydroData Expert: Raw vs Quality Control**")
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
                if any(fuzz.partial_ratio(col_lower, w) > 75 or w in col_lower for w in words):
                    mapping[col] = data_type
                    break
        return mapping

    def apply_strict_qc(self, data, col_type):
        """HANYA untuk Cleaned Data. Raw Data tidak boleh lewat sini."""
        data_clean = data.copy()
        
        # Batas Logis Oseanografi (Hard Limits)
        limits = {
            'Salinitas': (0, 45),
            'Suhu': (0, 40),
            'Water Level': (-5, 15)
        }
        low, high = limits.get(col_type, (data.min(), data.max()))
        
        # 1. Buang error sensor (99.99 dll)
        data_clean[(data_clean > high) | (data_clean < low)] = np.nan
        
        # 2. Buang Spike Statistik (IQR)
        q1 = np.nanpercentile(data_clean, 25)
        q3 = np.nanpercentile(data_clean, 75)
        iqr = q3 - q1
        data_clean[(data_clean < (q1 - 2*iqr)) | (data_clean > (q3 + 2*iqr))] = np.nan
        
        # 3. Interpolasi Linear (Mengisi gap yang dibuang)
        return pd.Series(data_clean).interpolate(method='linear').bfill().ffill().values

# 3. DASHBOARD VISUALISASI
def create_expert_dashboard(df, selected_col, processor):
    col_type = processor.detected_columns.get(selected_col, "Data")
    
    # RAW DATA: 100% Asli dari file tanpa perubahan
    raw_data = df[selected_col].values 
    
    # CLEANED DATA: Proses QC ketat
    cleaned_data = processor.apply_strict_qc(raw_data, col_type)
    
    time_idx = np.arange(len(raw_data))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'ORIGINAL RAW DATA (Tanpa Olah)', 
            'CLEANED DATA (QC Result)', 
            'Zoom: Spike Detection (100 Data)', 
            'Noise/Residual Analysis'
        ),
        vertical_spacing=0.15
    )

    # Plot 1: Raw Data (Harus kelihatan spike 99.99-nya)
    fig.add_trace(go.Scatter(x=time_idx, y=raw_data, name='RAW (Original)', line=dict(color='red', width=1)), row=1, col=1)

    # Plot 2: Cleaned Data (Harus stabil di angka 30-an)
    fig.add_trace(go.Scatter(x=time_idx, y=cleaned_data, name='CLEANED (QC)', line=dict(color='green', width=1.5)), row=1, col=2)

    # Plot 3: Zoom Comparison (Overlay)
    zoom_n = min(len(raw_data), 150)
    fig.add_trace(go.Scatter(x=time_idx[:zoom_n], y=raw_data[:zoom_n], name='Raw Zoom', line=dict(color='red', width=1, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_idx[:zoom_n], y=cleaned_data[:zoom_n], name='QC Zoom', line=dict(color='green', width=2)), row=2, col=1)

    # Plot 4: Residuals (Berapa banyak 'sampah' yang dibuang)
    residual = raw_data - cleaned_data
    fig.add_trace(go.Scatter(x=time_idx, y=residual, name='Residual/Noise', line=dict(color='purple')), row=2, col=2)

    # Labels
    fig.update_yaxes(title_text=f"{selected_col} (Raw)", row=1, col=1)
    fig.update_yaxes(title_text=f"{selected_col} (Cleaned)", row=1, col=2)
    fig.update_layout(height=850, template="plotly_white", title=f"<b>QC Report: {selected_col}</b>")
    
    return fig, cleaned_data

def main():
    st.sidebar.header("📁 Data Input")
    file = st.sidebar.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx'])

    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df_num = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        
        processor = HydroDataProcessor(df_num)
        selected_col = st.sidebar.selectbox("Pilih Parameter:", df_num.columns)
        
        if selected_col:
            tab1, tab2 = st.tabs(["📊 Analisis Perbandingan", "📥 Export Hasil"])
            
            with tab1:
                fig, cleaned_res = create_expert_dashboard(df_num, selected_col, processor)
                st.plotly_chart(fig, use_container_width=True)
                
                # Report Logis
                st.warning(f"⚠️ **RAW DATA** menunjukkan nilai maks: {np.nanmax(df_num[selected_col]):.2f}. Jika ini 99.99, itu adalah spike.")
                st.success(f"✅ **CLEANED DATA** memiliki rata-rata: {np.mean(cleaned_res):.2f}. Nilai ini sudah aman di range oseanografi.")

            with tab2:
                df_final = df.copy()
                df_final[f"{selected_col}_QC"] = cleaned_res
                st.download_button("💾 Download Hasil QC (.csv)", df_final.to_csv(index=False), "data_bersih.csv")
                st.dataframe(df_final.head(20))
    else:
        st.info("Upload file untuk melihat perbandingan Raw vs QC.")

if __name__ == "__main__":
    main()
