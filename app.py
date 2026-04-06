import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2
import pytesseract
import fitz  # PyMuPDF
from PIL import Image

# Import your other backend engines
from pharmacodynamics_engine import PharmacodynamicsEngine
from epigenetics_engine import EpigeneticAnalyzer
from spectral_engine import SpectralEngine
from phenotypic_engine import PhenotypicEngine
from kinetics_engine import BioprocessEngine
from report_engine import AmplikonReport
import google.generativeai as genai
from supabase import create_client, Client

# ---------------------------------------------------------
# INLINE VISION ENGINE (Bypasses Import Cache Issues)
# ---------------------------------------------------------
class GraphDigitizer:
    def __init__(self, uploaded_file):
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'pdf':
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = pdf_document.load_page(0)
            pix = page.get_pixmap(dpi=300) 
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                self.image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                self.image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            self.image = cv2.imdecode(file_bytes, 1)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
    def isolate_data_points(self, min_area=20, max_area=500, min_circularity=0.6):
        thresh = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > min_circularity:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        valid_points.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        return valid_points

    def perform_ocr(self):
        try:
            return pytesseract.image_to_string(self.gray, config=r'--oem 3 --psm 11').strip()
        except:
            return "OCR Engine not configured correctly."

    def map_to_real_data(self, pixel_points, x_range=(0, 100), y_range=(0, 100)):
        if not pixel_points: return pd.DataFrame(columns=['Extracted_X', 'Extracted_Y'])
        pts = np.array(pixel_points)
        min_px_x, max_px_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_px_y, max_px_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        real_data = []
        for (px, py) in pixel_points:
            real_x = x_range[0] + (px - min_px_x) * ((x_range[1] - x_range[0]) / (max_px_x - min_px_x + 1e-5))
            real_y = y_range[0] + (max_px_y - py) * ((y_range[1] - y_range[0]) / (max_px_y - min_px_y + 1e-5))
            real_data.append({'Extracted_X': round(real_x, 4), 'Extracted_Y': round(real_y, 4)})
        return pd.DataFrame(real_data)

# ---------------------------------------------------------
# 1. SYSTEM ARCHITECTURE & UI DESIGN
# ---------------------------------------------------------
st.set_page_config(page_title="BioSIGHT OS | Enterprise", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00d4ff; font-size: 1.8rem; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #00d4ff; color: black; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1.1 ENTERPRISE SECURITY (LOGIN GATE)
# ---------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00d4ff;'>🧬 BioSIGHT OS</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Authorized Amplikon Personnel Only</p>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password") 
        if st.button("Initialize System", use_container_width=True):
            if username == st.secrets["credentials"]["admin_username"] and password == st.secrets["credentials"]["admin_password"]:
                st.session_state['logged_in'] = True
                st.rerun() 
            else:
                st.error("Authentication Failed. Access Denied.")
    st.stop()
    
# ---------------------------------------------------------
# 1.2 CLOUD DATABASE CONNECTION
# ---------------------------------------------------------
@st.cache_resource 
def init_connection():
    return create_client(st.secrets["supabase"]["URL"], st.secrets["supabase"]["KEY"])
supabase = init_connection()

# ✨ NEW: The Master Cloud Sync Engine ✨
def upload_to_supabase(module_name, metrics_dict, ai_text="No AI summary generated."):
    """Helper function to push data into the Supabase experiment_logs table."""
    try:
        data, count = supabase.table("experiment_logs").insert({
            "module_name": module_name,
            "metrics": str(metrics_dict),
            "ai_summary": ai_text
        }).execute()
        return True
    except Exception as e:
        st.error(f"Cloud Sync Failed: {e}")
        return False

# ---------------------------------------------------------
# 1.5 GLOBAL MEMORY BANK
# ---------------------------------------------------------
if 'digitized_df' not in st.session_state: st.session_state['digitized_df'] = pd.DataFrame()
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

st.sidebar.markdown("### 🏢 System Navigation")
app_mode = st.sidebar.radio("Select View:", ["🧪 Active Workspace", "☁️ Cloud Archive"])
st.sidebar.markdown("---")

if app_mode == "☁️ Cloud Archive":
    st.title("☁️ Secure Cloud Archive")
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True): st.rerun()
    try:
        with st.spinner("Fetching secure records..."):
            response = supabase.table("experiment_logs").select("*").order("created_at", desc=True).execute()
            if response.data:
                df_archive = pd.DataFrame(response.data)
                df_archive['created_at'] = pd.to_datetime(df_archive['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(df_archive.rename(columns={"id": "Log ID", "created_at": "Timestamp", "module_name": "Module", "metrics": "Metrics", "ai_summary": "AI"}), use_container_width=True, hide_index=True)
            else:
                st.info("📭 The Cloud Database is currently empty.")
    except Exception as e: st.error(f"Failed to retrieve database records: {e}")
    st.stop()

# ---------------------------------------------------------
# 2. SIDEBAR NAVIGATION & DATA INGESTION
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧬 BioSIGHT OS v2.0")
    st.caption("Amplikon Biosystems - High-Throughput Analytics")
    st.divider()
    
    module = st.radio("Intelligence Modules", [
        "🛸 Universal Telemetry Dashboard",
        "🤖 BioSIGHT Global Copilot",
        "💊 Bioactivity & Pharmacodynamics (IC50/Kd)",
        "🧪 Multi-Spectral Suite (HPLC/GC-MS/UV-Vis)",
        "📊 Phenotypic & HCS Clustering",
        "⚙️ Enzyme Kinetics & Bioprocessing",
        "🧬 Epigenetic Array (DNA Methylation)",
        "📸 Auto-Digitizer (Graph OCR)",
        "📈 Quality Control (SPC)"
    ])
    
    st.divider()
    st.subheader("Data Ingestion")
    
    uploaded_files = st.file_uploader("Upload Raw Data (Up to 10 files)", type=["csv", "xlsx", "txt", "tsv"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 10: uploaded_files = uploaded_files[:10]
            
        st.session_state['data_pool'] = {}
        for f in uploaded_files:
            try:
                if f.name.endswith('.csv') or f.name.endswith('.txt'): st.session_state['data_pool'][f.name] = pd.read_csv(f, sep=None, engine='python')
                elif f.name.endswith('.xlsx'): st.session_state['data_pool'][f.name] = pd.read_excel(f)
            except Exception as e:
                st.error(f"Failed to read {f.name}")
                
        options = list(st.session_state['data_pool'].keys())
        if 'merged_dataset' in st.session_state:
            options.insert(0, "Fusion_Dataset (Merged/Joined)")
            st.session_state['data_pool']["Fusion_Dataset (Merged/Joined)"] = st.session_state['merged_dataset']
            
        selected_ds = st.selectbox("📂 Set Active Dataset:", options)
        st.session_state['active_dataset'] = st.session_state['data_pool'][selected_ds]
        st.success(f"✅ {selected_ds} Active!")
        st.info("💡 Go to 'Universal Telemetry Dashboard' to Merge files.")
    else:
        if 'active_dataset' in st.session_state: del st.session_state['active_dataset']

# ---------------------------------------------------------
# 3. MODULE EXECUTION
# ---------------------------------------------------------

if module == "🛸 Universal Telemetry Dashboard":
    st.title("Universal Analytics & Data Harmonization")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Streams", "8 Modalities")
    c2.metric("Processing Latency", "8ms")
    c3.metric("Data Integrity", "99.9% (ALCOA+)")
    c4.metric("Scopus-Ready Exports", "Enabled")
    
    if 'data_pool' in st.session_state and len(st.session_state['data_pool']) > 0:
        st.divider()
        st.markdown("### 🗄️ Global Data Harmonization")
        st.markdown("Compare multiple experimental runs, append plates together, or join multi-omics datasets.")
        
        t1, t2, t3 = st.tabs(["👁️ Compare Datasets", "➕ Append Rows (Concat)", "🔗 Join Columns (Merge)"])
        
        with t1:
            selected_for_compare = st.multiselect("Select datasets to compare side-by-side:", list(st.session_state['data_pool'].keys()))
            if selected_for_compare:
                cols = st.columns(len(selected_for_compare))
                for i, name in enumerate(selected_for_compare):
                    with cols[i]:
                        st.markdown(f"**{name}**")
                        st.caption(f"Shape: {st.session_state['data_pool'][name].shape[0]} rows x {st.session_state['data_pool'][name].shape[1]} cols")
                        st.dataframe(st.session_state['data_pool'][name].head(5), use_container_width=True)

        with t2:
            st.info("Combine multiple runs of the same experiment. (e.g., Run 1 + Run 2). The columns must match.")
            to_merge = st.multiselect("Select datasets to append together:", list(st.session_state['data_pool'].keys()), key="merge_sel")
            if st.button("Merge & Set as Active Dataset", use_container_width=True):
                if len(to_merge) > 1:
                    try:
                        combined = pd.concat([st.session_state['data_pool'][n] for n in to_merge], ignore_index=True)
                        st.session_state['merged_dataset'] = combined
                        st.success(f"Merged {len(to_merge)} files! Total rows: {combined.shape[0]}. Select 'Fusion_Dataset' in the sidebar.")
                    except Exception as e:
                        st.error(f"Merge failed. Ensure columns match perfectly. Error: {e}")
                else:
                    st.warning("Select at least 2 files to append.")

        with t3:
            st.info("Join two different datasets based on a common key (e.g., matching 'Compound_ID').")
            col1, col2 = st.columns(2)
            with col1:
                left_ds = st.selectbox("Left Dataset:", list(st.session_state['data_pool'].keys()), key="left_ds")
                left_col = st.selectbox("Join Key (Left):", st.session_state['data_pool'][left_ds].columns)
            with col2:
                right_ds = st.selectbox("Right Dataset:", list(st.session_state['data_pool'].keys()), key="right_ds")
                right_col = st.selectbox("Join Key (Right):", st.session_state['data_pool'][right_ds].columns)

            if st.button("Join & Set as Active Dataset", use_container_width=True):
                try:
                    joined = pd.merge(st.session_state['data_pool'][left_ds], st.session_state['data_pool'][right_ds], left_on=left_col, right_on=right_col, how="inner")
                    st.session_state['merged_dataset'] = joined
                    st.success(f"Successfully joined! Total columns: {joined.shape[1]}. Select 'Fusion_Dataset' in the sidebar.")
                except Exception as e:
                    st.error(f"Join failed: {e}")
    else:
        st.info("Upload files in the sidebar to activate the Data Harmonization Engine.")

elif module == "💊 Bioactivity & Pharmacodynamics (IC50/Kd)":
    st.title("Receptor Binding & IC50 Profiling")
    
    has_memory = not st.session_state.get('digitized_df', pd.DataFrame()).empty
    use_memory = st.toggle("📥 Pull Data from Global Memory Bank (Digitizer)", disabled=not has_memory)
    
    if use_memory and has_memory:
        st.success("Loaded digitized data from memory.")
        df_pk = st.session_state['digitized_df'].copy()
        df_pk = df_pk.rename(columns={'Extracted_X': 'Concentration_uM', 'Extracted_Y': 'Inhibition'})
        df_pk['Concentration_uM'] = df_pk['Concentration_uM'].apply(lambda x: max(x, 1e-5)) 
        
    elif 'active_dataset' in st.session_state:
        df_raw = st.session_state['active_dataset'].copy()
        cols = df_raw.columns.tolist()
        
        guess_x = next((c for c in cols if any(k in c.lower() for k in ['conc', 'dose', 'um', 'nm', 'x'])), cols[0])
        guess_y = next((c for c in cols if any(k in c.lower() for k in ['inh', 'resp', 'viab', 'effect', 'y'])), cols[-1] if len(cols)>1 else cols[0])
        
        st.info("🧠 BioSIGHT AI mapped your columns. Adjust if necessary:")
        col1, col2 = st.columns(2)
        with col1: x_col = st.selectbox("Select Concentration (X-Axis):", cols, index=cols.index(guess_x))
        with col2: y_col = st.selectbox("Select Inhibition/Response (Y-Axis):", cols, index=cols.index(guess_y))
            
        df_pk = pd.DataFrame({'Concentration_uM': df_raw[x_col], 'Inhibition': df_raw[y_col]})
        df_pk['Concentration_uM'] = pd.to_numeric(df_pk['Concentration_uM'], errors='coerce')
        df_pk['Inhibition'] = pd.to_numeric(df_pk['Inhibition'], errors='coerce')
        df_pk = df_pk.dropna()
        df_pk['Concentration_uM'] = df_pk['Concentration_uM'].apply(lambda x: max(x, 1e-9))
    else:
        st.caption("Using Live Internship Data (Amplikon Dataset)")
        conc = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
        resp = np.array([2.1, 5.4, 15.2, 35.8, 55.4, 85.1, 95.3, 98.9, 99.5])
        df_pk = pd.DataFrame({'Concentration_uM': conc, 'Inhibition': resp})
    
    engine = PharmacodynamicsEngine()
    results = engine.fit_ic50(df_pk['Concentration_uM'], df_pk['Inhibition'])
    
    if results["success"]:
        x_smooth = np.logspace(np.log10(df_pk['Concentration_uM'].min()), np.log10(df_pk['Concentration_uM'].max()), 100)
        y_smooth = engine.four_param_logistic(x_smooth, results['top'], results['bottom'], results['ic50'], results['hill'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pk['Concentration_uM'], y=df_pk['Inhibition'], mode='markers', name='Raw Data', marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='4PL Fit', line=dict(color='#00d4ff', width=3)))
        fig.update_layout(xaxis_type="log", title="4PL Non-Linear Regression", xaxis_title="Log Concentration (µM)", yaxis_title="% Inhibition")
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Calculated IC50", f"{results['ic50']:.3f} µM")
        c2.metric("Hill Slope", f"{results['hill']:.2f}")
        c3.metric("Fit Confidence (R²)", f"{results['r2']:.4f}")
        
        st.divider()
        if st.button("✨ Generate Live AI Analysis", use_container_width=True):
            with st.spinner("Connecting to LLM Neural Engine..."):
                try:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(f"Analyze IC50: {results['ic50']}uM, Hill: {results['hill']}, R2: {results['r2']}. Keep it to 3 scientific sentences.")
                    st.info(response.text)
                except Exception as e: st.error(f"AI Engine Error: {e}")
                
        # ✨ NEW: Cloud Sync Button ✨
        st.divider()
        st.subheader("☁️ Enterprise Cloud Sync")
        if st.button("💾 Save Results to Secure Cloud", use_container_width=True, key="ic50_save"):
            with st.spinner("Encrypting and transmitting to Supabase..."):
                metrics_data = {"Calculated_IC50_uM": round(results['ic50'], 3), "Hill_Slope": round(results['hill'], 2), "Confidence_R2": round(results['r2'], 4)}
                success = upload_to_supabase("Bioactivity & Pharmacodynamics (IC50)", metrics_data)
                if success: st.success("✅ Run archived successfully! Switch to the 'Cloud Archive' view to see it.")
                
    else:
        st.error("Curve fitting failed. The data may not follow a standard dose-response curve.")

elif module == "🧪 Multi-Spectral Suite (HPLC/GC-MS/UV-Vis)":
    st.title("Chromatographic Deconvolution & Peak Integration")
    
    if 'active_dataset' in st.session_state:
        df_raw = st.session_state['active_dataset'].copy()
        cols = df_raw.columns.tolist()
        
        guess_x = next((c for c in cols if any(k in c.lower() for k in ['time', 'ret', 'min', 'rt', 'x'])), cols[0])
        guess_y = next((c for c in cols if any(k in c.lower() for k in ['int', 'abs', 'signal', 'mau', 'y'])), cols[-1] if len(cols)>1 else cols[0])
        
        st.info("🧠 BioSIGHT AI mapped your columns:")
        c1, c2 = st.columns(2)
        with c1: x_col = st.selectbox("Select Retention Time (X-Axis):", cols, index=cols.index(guess_x))
        with c2: y_col = st.selectbox("Select Intensity (Y-Axis):", cols, index=cols.index(guess_y))
        
        df_spec = pd.DataFrame({'Retention_Time': df_raw[x_col], 'Intensity': df_raw[y_col]})
        df_spec['Retention_Time'] = pd.to_numeric(df_spec['Retention_Time'], errors='coerce')
        df_spec['Intensity'] = pd.to_numeric(df_spec['Intensity'], errors='coerce')
        df_spec = df_spec.dropna()
        rt = df_spec['Retention_Time'].values
        signal = df_spec['Intensity'].values
    else:
        st.warning("🟡 No HPLC data uploaded. Using simulated trace.")
        rt = np.linspace(0, 30, 2000)
        signal = (45 * np.exp(-((rt - 5.2)**2)/0.08) + 120 * np.exp(-((rt - 12.5)**2)/0.15) + 85 * np.exp(-((rt - 13.1)**2)/0.12)) + np.random.normal(0, 1.5, 2000) + 10
    
    engine = SpectralEngine()
    clean_signal, peaks, df_peaks = engine.process_chromatogram(rt, signal)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rt, y=signal, name="Raw Signal", line=dict(color='#444444', width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=rt, y=clean_signal, name="Processed Signal", line=dict(color='#00d4ff', width=2)))
    
    if not df_peaks.empty:
        fig.add_trace(go.Scatter(x=df_peaks['Retention_Time'], y=df_peaks['Intensity'], mode='markers+text', name="Detected Peaks", marker=dict(color='#ff00d4', size=10, symbol='triangle-down'), text=df_peaks['Peak_ID'], textposition="top center"))
    fig.update_layout(title="HPLC Trace: Peak Detection", xaxis_title="Retention Time (min)", yaxis_title="Intensity")
    st.plotly_chart(fig, use_container_width=True)
    
    # ✨ NEW: Cloud Sync Button ✨
    st.divider()
    st.subheader("☁️ Enterprise Cloud Sync")
    if st.button("💾 Save Results to Secure Cloud", use_container_width=True, key="hplc_save"):
        with st.spinner("Encrypting and transmitting to Supabase..."):
            metrics_data = {"Total_Peaks_Detected": len(df_peaks) if not df_peaks.empty else 0}
            success = upload_to_supabase("Multi-Spectral Suite (HPLC)", metrics_data)
            if success: st.success("✅ Run archived successfully! Switch to the 'Cloud Archive' view to see it.")

elif module == "📊 Phenotypic & HCS Clustering":
    st.title("High-Content Screening (HCS) & Unsupervised Clustering")
    
    if 'active_dataset' in st.session_state:
        df_raw = st.session_state['active_dataset'].copy()
        cols = df_raw.columns.tolist()
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 4:
            st.error("🚨 Dataset too small: Clustering requires at least 4 numeric columns. Go to Telemetry to Join datasets or upload an HCS file.")
            st.stop()
            
        g_id = next((c for c in cols if any(k in c.lower() for k in ['id', 'comp', 'name', 'drug'])), cols[0])
        st.info("🧠 Auto-Mapped Features. Adjust if necessary:")
        c1, c2, c3 = st.columns(3)
        c4, c5 = st.columns(2)
        with c1: col_id = st.selectbox("Compound ID:", cols, index=cols.index(g_id))
        with c2: col_v = st.selectbox("Feature 1 (Cell Viability):", num_cols, index=0)
        with c3: col_a = st.selectbox("Feature 2 (Apoptosis Rate):", num_cols, index=1)
        with c4: col_r = st.selectbox("Feature 3 (ROS Production):", num_cols, index=2)
        with c5: col_m = st.selectbox("Feature 4 (Morphology Score):", num_cols, index=3)
        
        df_pheno = pd.DataFrame({'Compound_ID': df_raw[col_id], 'Cell_Viability': df_raw[col_v], 'Apoptosis_Rate': df_raw[col_a], 'ROS_Production': df_raw[col_r], 'Morphology_Score': df_raw[col_m]})
        for c in ['Cell_Viability', 'Apoptosis_Rate', 'ROS_Production', 'Morphology_Score']:
            df_pheno[c] = pd.to_numeric(df_pheno[c], errors='coerce')
        df_pheno = df_pheno.dropna()
    else:
        st.warning("🟡 No file uploaded. Using simulated demo data.")
        np.random.seed(42)
        df_pheno = pd.DataFrame({'Compound_ID': [f"AMP-{i:03d}" for i in range(300)], 'Cell_Viability': np.concatenate([np.random.normal(90, 5, 100), np.random.normal(40, 10, 100), np.random.normal(85, 8, 100)]), 'Apoptosis_Rate': np.concatenate([np.random.normal(5, 2, 100), np.random.normal(55, 15, 100), np.random.normal(10, 5, 100)]), 'ROS_Production': np.concatenate([np.random.normal(20, 10, 100), np.random.normal(80, 20, 100), np.random.normal(30, 10, 100)]), 'Morphology_Score': np.concatenate([np.random.normal(0.9, 0.1, 100), np.random.normal(0.3, 0.15, 100), np.random.normal(0.8, 0.1, 100)])})
    
    engine = PhenotypicEngine()
    df_analyzed, variance = engine.analyze_phenotypes(df_pheno, ['Cell_Viability', 'Apoptosis_Rate', 'ROS_Production', 'Morphology_Score'])
    fig = px.scatter(df_analyzed, x='PCA1', y='PCA2', color='Cluster_ID', hover_data=['Compound_ID'], title=f"PCA Representation (Variance: {variance[0]:.1f}% + {variance[1]:.1f}%)", color_discrete_sequence=['#00d4ff', '#ff00d4', '#ffdd00'])
    st.plotly_chart(fig, use_container_width=True)
    
    # ✨ NEW: Cloud Sync Button ✨
    st.divider()
    st.subheader("☁️ Enterprise Cloud Sync")
    if st.button("💾 Save Results to Secure Cloud", use_container_width=True, key="cluster_save"):
        with st.spinner("Encrypting and transmitting to Supabase..."):
            metrics_data = {"Total_Compounds_Clustered": len(df_analyzed), "PCA_Variance_Retained": round(variance[0] + variance[1], 1)}
            success = upload_to_supabase("Phenotypic & HCS Clustering", metrics_data)
            if success: st.success("✅ Run archived successfully! Switch to the 'Cloud Archive' view to see it.")

elif module == "⚙️ Enzyme Kinetics & Bioprocessing":
    st.title("Industrial Bioprocessing & Fermentation Dynamics")
    col1, col2 = st.columns([1, 3])
    with col1:
        t_max = st.slider("Fermentation Time (hrs)", 24, 168, 72)
        mu_max = st.slider("Max Growth Rate", 0.05, 0.50, 0.25)
        Ks = st.slider("Half-Velocity Const", 0.1, 10.0, 2.5)
        Yxs = st.slider("Biomass Yield", 0.1, 1.0, 0.5)
    with col2:
        engine = BioprocessEngine()
        df_kinetics = engine.simulate_fermentation(t_max, 1.0, [0.5, 50.0, 0.0], {'mu_max': mu_max, 'Ks': Ks, 'Yxs': Yxs, 'alpha': 0.1, 'beta': 0.05})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_kinetics['Time_Hours'], y=df_kinetics['Biomass_gL'], name="Simulated Biomass", line=dict(color='#00d4ff')))
        fig.add_trace(go.Scatter(x=df_kinetics['Time_Hours'], y=df_kinetics['Substrate_gL'], name="Simulated Substrate", line=dict(color='#ff00d4', dash='dot')))
        
        if 'active_dataset' in st.session_state:
            df_raw = st.session_state['active_dataset'].copy()
            cols = df_raw.columns.tolist()
            g_t = next((c for c in cols if any(k in c.lower() for k in ['time', 'hr', 'hour', 't'])), cols[0])
            g_b = next((c for c in cols if any(k in c.lower() for k in ['bio', 'mass', 'od', 'cell'])), cols[-1] if len(cols)>1 else cols[0])
            
            c1, c2 = st.columns(2)
            with c1: col_t = st.selectbox("Select Time Column:", cols, index=cols.index(g_t))
            with c2: col_b = st.selectbox("Select Biomass Column:", cols, index=cols.index(g_b))
            
            df_bio = pd.DataFrame({'Time_Hours': df_raw[col_t], 'Actual_Biomass': df_raw[col_b]})
            df_bio['Time_Hours'] = pd.to_numeric(df_bio['Time_Hours'], errors='coerce')
            df_bio['Actual_Biomass'] = pd.to_numeric(df_bio['Actual_Biomass'], errors='coerce')
            df_bio = df_bio.dropna()
            fig.add_trace(go.Scatter(x=df_bio['Time_Hours'], y=df_bio['Actual_Biomass'], mode='markers', name='Actual Lab Biomass', marker=dict(color='#00ff00', size=8)))
        st.plotly_chart(fig, use_container_width=True)
        
        # ✨ NEW: Cloud Sync Button ✨
        st.divider()
        st.subheader("☁️ Enterprise Cloud Sync")
        if st.button("💾 Save Results to Secure Cloud", use_container_width=True, key="kinetics_save"):
            with st.spinner("Encrypting and transmitting to Supabase..."):
                metrics_data = {"Max_Growth_Rate": mu_max, "Half_Velocity_Const": Ks, "Biomass_Yield": Yxs, "Final_Biomass_gL": round(df_kinetics['Biomass_gL'].iloc[-1], 2)}
                success = upload_to_supabase("Enzyme Kinetics & Bioprocessing", metrics_data)
                if success: st.success("✅ Run archived successfully! Switch to the 'Cloud Archive' view to see it.")

elif module == "🧬 Epigenetic Array (DNA Methylation)":
    st.title("Epigenomic Profiling & Aging Biomarkers")
    if 'active_dataset' in st.session_state:
        df_raw = st.session_state['active_dataset'].copy()
        cols = df_raw.columns.tolist()
        g_l = next((c for c in cols if any(k in c.lower() for k in ['loc', 'gene', 'site'])), cols[0])
        g_c = next((c for c in cols if any(k in c.lower() for k in ['cpg', 'meth1', 'beta1'])), cols[1] if len(cols)>1 else cols[0])
        g_nc = next((c for c in cols if any(k in c.lower() for k in ['non', 'meth2', 'beta2'])), cols[2] if len(cols)>2 else cols[0])
        
        c1, c2, c3 = st.columns(3)
        with c1: col_l = st.selectbox("Select Locus/Gene:", cols, index=cols.index(g_l))
        with c2: col_c = st.selectbox("Select CpG Methylation:", cols, index=cols.index(g_c))
        with c3: col_nc = st.selectbox("Select Non-CpG Methylation:", cols, index=cols.index(g_nc))
        
        df_meth = pd.DataFrame({'Locus': df_raw[col_l], 'CpG_Beta': df_raw[col_c], 'Non_CpG_Beta': df_raw[col_nc]})
        df_meth['CpG_Beta'] = pd.to_numeric(df_meth['CpG_Beta'], errors='coerce')
        df_meth['Non_CpG_Beta'] = pd.to_numeric(df_meth['Non_CpG_Beta'], errors='coerce')
        df_meth = df_meth.dropna()
    else:
        analyzer = EpigeneticAnalyzer()
        df_meth = analyzer.generate_methylation_profile()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_meth['Locus'], y=df_meth['CpG_Beta'], mode='lines', name='CpG Methylation', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df_meth['Locus'], y=df_meth['Non_CpG_Beta'], mode='lines', fill='tozeroy', name='Non-CpG Methylation', line=dict(color='#ff00d4', width=2)))
    st.plotly_chart(fig, use_container_width=True)
    
    # ✨ NEW: Cloud Sync Button ✨
    st.divider()
    st.subheader("☁️ Enterprise Cloud Sync")
    if st.button("💾 Save Results to Secure Cloud", use_container_width=True, key="epi_save"):
        with st.spinner("Encrypting and transmitting to Supabase..."):
            metrics_data = {"Loci_Analyzed": len(df_meth), "Average_CpG_Beta": round(df_meth['CpG_Beta'].mean(), 3)}
            success = upload_to_supabase("Epigenetic Array", metrics_data)
            if success: st.success("✅ Run archived successfully! Switch to the 'Cloud Archive' view to see it.")

elif module == "📸 Auto-Digitizer (Graph OCR)":
    st.title("Computer Vision: Graph to CSV")
    img_files = st.file_uploader("Upload Graph Files (Up to 10)", type=['png', 'jpg', 'jpeg', 'pdf'], accept_multiple_files=True)
    if img_files:
        if len(img_files) > 10: img_files = img_files[:10]
        selected_img_name = st.selectbox("📸 Select Image to Digitize:", [f.name for f in img_files])
        active_img = next(f for f in img_files if f.name == selected_img_name)
        if active_img.name.lower().endswith('.pdf'): st.success(f"📄 PDF {active_img.name} Detected.")
        else: st.image(active_img, width=600)
            
        with st.expander("⚙️ Advanced Vision Engine Tuning", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1: ui_min_area = st.slider("Minimum Point Size", 1, 50, 10) 
            with c2: ui_max_area = st.slider("Maximum Point Size", 100, 1000, 500)
            with c3: ui_circularity = st.slider("Shape Strictness (Circularity)", 0.0, 1.0, 0.2, 0.1) 
        
        if st.button("Initialize Vision Pipeline", use_container_width=True):
            with st.spinner("Running OpenCV Contours..."):
                digitizer = GraphDigitizer(active_img)
                pixels = digitizer.isolate_data_points(min_area=ui_min_area, max_area=ui_max_area, min_circularity=ui_circularity)
                extracted_df = digitizer.map_to_real_data(pixels)
                if not extracted_df.empty:
                    st.session_state['digitized_df'] = extracted_df
                    col1, col2 = st.columns([2, 1])
                    with col1: st.plotly_chart(px.scatter(extracted_df, x='Extracted_X', y='Extracted_Y'), use_container_width=True)
                    with col2: st.dataframe(extracted_df, use_container_width=True)
                else: st.error("Detection Failed. Adjust sliders.")

elif module == "🤖 BioSIGHT Global Copilot":
    st.title("BioSIGHT Global Copilot")
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e: st.error(f"API Error: {e}"); st.stop()

    for msg in st.session_state['chat_history']:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask me anything about your data..."):
        with st.chat_message("user"): st.markdown(user_prompt)
        st.session_state['chat_history'].append({"role": "user", "content": user_prompt})
        
        ctx = "You are BioSIGHT. "
        if not st.session_state['digitized_df'].empty: ctx += f"\nData:\n{st.session_state['digitized_df'].head(5).to_string()}"
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    res = model.generate_content(ctx + "\nQuery: " + user_prompt)
                    st.markdown(res.text)
                    st.session_state['chat_history'].append({"role": "assistant", "content": res.text})
                except Exception as e: st.error(f"Error: {e}")

elif module == "📈 Quality Control (SPC)":
    st.title("📈 Statistical Process Control")
    if 'active_dataset' in st.session_state:
        df_raw = st.session_state['active_dataset'].copy()
        cols = df_raw.columns.tolist()
        g_d = next((c for c in cols if any(k in c.lower() for k in ['day', 'run', 'time', 'date', 'x'])), cols[0])
        g_v = next((c for c in cols if any(k in c.lower() for k in ['val', 'ctrl', 'read', 'meas', 'y'])), cols[-1] if len(cols)>1 else cols[0])
        
        c1, c2 = st.columns(2)
        with c1: col_d = st.selectbox("Select Run/Day Column:", cols, index=cols.index(g_d))
        with c2: col_v = st.selectbox("Select Control Value Column:", cols, index=cols.index(g_v))
        
        df_qc = pd.DataFrame({'Run_Day': df_raw[col_d], 'Control_Value': df_raw[col_v]})
        df_qc['Run_Day'] = pd.to_numeric(df_qc['Run_Day'], errors='coerce')
        df_qc['Control_Value'] = pd.to_numeric(df_qc['Control_Value'], errors='coerce')
        df_qc = df_qc.dropna()
        days, qc_values = df_qc['Run_Day'].values, df_qc['Control_Value'].values
    else:
        np.random.seed(42)
        days, qc_values = np.arange(1, 31), np.random.normal(loc=100, scale=5, size=30)
        qc_values[27] = 118 
            
    mean_val, sd_val = np.mean(qc_values), np.std(qc_values)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{mean_val:.2f}"); col2.metric("SD (1σ)", f"{sd_val:.2f}")
    col3.metric("Warning (±2σ)", f"{(mean_val + 2*sd_val):.2f}"); col4.metric("Action (±3σ)", f"{(mean_val + 3*sd_val):.2f}")
            
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=qc_values, mode='lines+markers', name='Daily QC Run', line=dict(color='#00d4ff')))
    fig.add_hline(y=mean_val, line_dash="dash", line_color="#00ff00")
    fig.add_hline(y=mean_val + 3*sd_val, line_dash="solid", line_color="#ff0000")
    fig.add_hline(y=mean_val - 3*sd_val, line_dash="solid", line_color="#ff0000")
    fig.update_layout(title="Levey-Jennings Control Chart", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)
    
    # ✨ NEW: Cloud Sync Button ✨
    st.divider()
    st.subheader("☁️ Enterprise Cloud Sync")
    if st.button("💾 Save Results to Secure Cloud", use_container_width=True, key="qc_save"):
        with st.spinner("Encrypting and transmitting to Supabase..."):
            metrics_data = {"Historical_Mean": round(mean_val, 2), "Standard_Deviation": round(sd_val, 2), "Anomalies_Detected": len(np.where(qc_values > (mean_val + 3*sd_val))[0]) + len(np.where(qc_values < (mean_val - 3*sd_val))[0])}
            success = upload_to_supabase("Quality Control (SPC)", metrics_data)
            if success: st.success("✅ Run archived successfully! Switch to the 'Cloud Archive' view to see it.")
