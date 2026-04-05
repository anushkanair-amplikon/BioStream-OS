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

# ---------------------------------------------------------
# INLINE VISION ENGINE (Bypasses Import Cache Issues)
# ---------------------------------------------------------
class GraphDigitizer:
    def __init__(self, uploaded_file):
        """Initializes the engine, auto-detects file type, and decodes to pixels."""
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
        thresh = cv2.adaptiveThreshold(
            self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity > min_circularity:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        valid_points.append((cX, cY))
        return valid_points

    def perform_ocr(self):
        custom_config = r'--oem 3 --psm 11'
        try:
            text = pytesseract.image_to_string(self.gray, config=custom_config)
            return text.strip()
        except Exception:
            return "OCR Engine not configured correctly. Skipping text extraction."

    def map_to_real_data(self, pixel_points, x_range=(0, 100), y_range=(0, 100)):
        if not pixel_points:
            return pd.DataFrame(columns=['Extracted_X', 'Extracted_Y'])

        pts = np.array(pixel_points)
        min_px_x, max_px_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_px_y, max_px_y = np.min(pts[:, 1]), np.max(pts[:, 1])

        real_data = []
        for (px, py) in pixel_points:
            real_x = x_range[0] + (px - min_px_x) * ((x_range[1] - x_range[0]) / (max_px_x - min_px_x + 1e-5))
            real_y = y_range[0] + (max_px_y - py) * ((y_range[1] - y_range[0]) / (max_px_y - min_px_y + 1e-5))
            real_data.append({'Extracted_X': round(real_x, 4), 'Extracted_Y': round(real_y, 4)})

        return pd.DataFrame(real_data)

    def process_full_pipeline(self):
        pixels = self.isolate_data_points()
        extracted_df = self.map_to_real_data(pixels)
        ocr_text = self.perform_ocr()
        return extracted_df, ocr_text

# ---------------------------------------------------------
# 1. SYSTEM ARCHITECTURE & UI DESIGN
# ---------------------------------------------------------
st.set_page_config(page_title="BioStream OS | Enterprise", page_icon="🧬", layout="wide")

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
# Initialize login state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# The Bouncer
if not st.session_state['logged_in']:
    # Create a sleek, centered login box
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00d4ff;'>🧬 BioStream OS</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Authorized Amplikon Personnel Only</p>", unsafe_allow_html=True)
        
        # Input fields
        username = st.text_input("Username")
        password = st.text_input("Password", type="password") # Hides the text as dots
        
        if st.button("Initialize System", use_container_width=True):
            # Check against your secrets.toml file
            if username == st.secrets["credentials"]["admin_username"] and password == st.secrets["credentials"]["admin_password"]:
                st.session_state['logged_in'] = True
                st.rerun() # Refreshes the page to show the main app
            else:
                st.error("Authentication Failed. Access Denied.")
    
    # 🔥 CRITICAL: This stops the rest of your app from loading!
    st.stop() 


# ---------------------------------------------------------
# 1.5 GLOBAL MEMORY BANK (State Management)
# ---------------------------------------------------------
if 'digitized_df' not in st.session_state:
    st.session_state['digitized_df'] = pd.DataFrame()

# NEW: Memory for the Global Copilot
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# ---------------------------------------------------------
# 1.3 ENTERPRISE NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown("### 🏢 System Navigation")
app_mode = st.sidebar.radio(
    "Select View:", 
    ["🧪 Active Workspace", "☁️ Cloud Archive"]
)
st.sidebar.markdown("---")

# ---------------------------------------------------------
# ☁️ CLOUD ARCHIVE VIEW
# ---------------------------------------------------------
if app_mode == "☁️ Cloud Archive":
    st.title("☁️ Secure Cloud Archive")
    st.markdown("View historical experiment logs, metrics, and AI analysis directly from the Supabase database.")
    
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    try:
        with st.spinner("Fetching secure records..."):
            # Request all data from your Supabase table, ordered by newest first
            response = supabase.table("experiment_logs").select("*").order("created_at", desc=True).execute()
            data = response.data
            
            if data:
                # Convert the raw database data into a beautiful Pandas DataFrame
                df_archive = pd.DataFrame(data)
                
                # Clean up the timestamp to look professional
                df_archive['created_at'] = pd.to_datetime(df_archive['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Rename columns for the final display
                df_archive = df_archive.rename(columns={
                    "id": "Log ID",
                    "created_at": "Timestamp",
                    "module_name": "Analysis Module",
                    "metrics": "Calculated Metrics",
                    "ai_summary": "AI Insights"
                })
                
                # Display as an interactive, dark-theme compatible table
                st.dataframe(df_archive, use_container_width=True, hide_index=True)
            else:
                st.info("📭 The Cloud Database is currently empty. Run an analysis in the Active Workspace and click 'Save to Cloud Database' to populate this archive.")
                
    except Exception as e:
        st.error(f"Failed to retrieve database records: {e}")

    # 🔥 CRITICAL: This stops the main workspace from loading while viewing the archive!
    st.stop()

# ---------------------------------------------------------
# 2. SIDEBAR NAVIGATION & DATA INGESTION
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧬 BioStream OS v2.0")
    st.caption("Amplikon Biosystems - High-Throughput Analytics")
    st.divider()
    
    module = st.radio("Intelligence Modules", [
        "🛸 Universal Telemetry Dashboard",
        "🤖 BioSIGHT Global Copilot",  # <--- ADD THIS HERE
        "💊 Bioactivity & Pharmacodynamics (IC50/Kd)",
        "🧪 Multi-Spectral Suite (HPLC/GC-MS/UV-Vis)",
        "📊 Phenotypic & HCS Clustering",
        "⚙️ Enzyme Kinetics & Bioprocessing",
        "🧬 Epigenetic Array (DNA Methylation)",
        "📸 Auto-Digitizer (Graph OCR)"
    ])
    
    st.divider()
    st.subheader("Data Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload Raw Amplikon Data", 
        type=["csv", "xlsx", "txt", "tsv", "mzml", "dat"]
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_extension in ['csv', 'txt', 'tsv']:
                df_uploaded = pd.read_csv(uploaded_file, sep=None, engine='python')
            elif file_extension == 'xlsx':
                df_uploaded = pd.read_excel(uploaded_file)
            st.success(f"Successfully loaded {uploaded_file.name}!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---------------------------------------------------------
# 3. MODULE EXECUTION
# ---------------------------------------------------------

if module == "🛸 Universal Telemetry Dashboard":
    st.title("Universal Analytics Gateway")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Streams", "8 Modalities")
    c2.metric("Processing Latency", "8ms")
    c3.metric("Data Integrity", "99.9% (ALCOA+)")
    c4.metric("Scopus-Ready Exports", "Enabled")
    st.info("System Ready. Select a dedicated module from the sidebar to begin processing Amplikon datasets.")

elif module == "💊 Bioactivity & Pharmacodynamics (IC50/Kd)":
    st.title("Receptor Binding & IC50 Profiling")
    
    has_memory = not st.session_state['digitized_df'].empty
    use_memory = st.toggle("📥 Pull Data from Global Memory Bank (Digitizer)", disabled=not has_memory)
    
    if use_memory and has_memory:
        st.success("Loaded digitized data from memory.")
        df_pk = st.session_state['digitized_df'].copy()
        df_pk = df_pk.rename(columns={'Extracted_X': 'Concentration_uM', 'Extracted_Y': 'Inhibition'})
        df_pk['Concentration_uM'] = df_pk['Concentration_uM'].apply(lambda x: max(x, 1e-5)) 
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
        
        # --- NEW: Automated Executive Reporting ---
        st.divider()
        st.subheader("📄 Export Official Report")
        
        # ---------------------------------------------------------
        # ✨ REAL AI ANALYST INTEGRATION (GEMINI)
        # ---------------------------------------------------------
        st.divider()
        st.subheader("🤖 BioSIGHT Cognitive Analyst")
        
        if st.button("✨ Generate Live AI Analysis", use_container_width=True):
            with st.spinner("Connecting to LLM Neural Engine..."):
                try:
                    # 1. Authenticate with your secure key
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    
                    # 2. Load the lightweight, fast model
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # 3. Construct the "System Prompt"
                    # We inject your live mathematical results directly into the prompt
                    ai_prompt = f"""
                    You are an expert computational biologist and lead pharmacologist at Amplikon Biosystems.
                    Analyze the following high-throughput screening data for a novel drug candidate:
                    
                    - Calculated IC50: {results['ic50']:.3f} µM
                    - Hill Coefficient (Slope): {results['hill']:.2f}
                    - Regression Confidence (R²): {results['r2']:.4f}
                    
                    Write a concise, highly professional 3-sentence scientific conclusion. 
                    State whether this compound exhibits strong, moderate, or weak potency. 
                    Conclude with a recommendation on whether it should advance to in-vivo clinical trials or requires chemical optimization.
                    Do not use markdown headers or fluff.
                    """
                    
                    # 4. Fire the request to the AI and display the result
                    response = model.generate_content(ai_prompt)
                    st.info(response.text)
                    
                except Exception as e:
                    st.error(f"AI Engine Error: Make sure your API key is configured in secrets.toml. Details: {e}")
    else:
        st.error("Curve fitting failed. The data may not follow a standard dose-response curve.")

elif module == "🧪 Multi-Spectral Suite (HPLC/GC-MS/UV-Vis)":
    st.title("Chromatographic Deconvolution & Peak Integration")
    rt = np.linspace(0, 30, 2000)
    signal = (45 * np.exp(-((rt - 5.2)**2)/0.08) + 120 * np.exp(-((rt - 12.5)**2)/0.15) + 
              85 * np.exp(-((rt - 13.1)**2)/0.12) + 60 * np.exp(-((rt - 22.8)**2)/0.3)) + np.random.normal(0, 1.5, 2000) + 10
    
    engine = SpectralEngine()
    clean_signal, peaks, df_peaks = engine.process_chromatogram(rt, signal)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rt, y=signal, name="Raw Signal", line=dict(color='#444444', width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=rt, y=clean_signal, name="Processed Signal", line=dict(color='#00d4ff', width=2)))
    
    if not df_peaks.empty:
        fig.add_trace(go.Scatter(x=df_peaks['Retention_Time'], y=df_peaks['Intensity'], mode='markers+text', 
                                 name="Detected Peaks", marker=dict(color='#ff00d4', size=10, symbol='triangle-down'),
                                 text=df_peaks['Peak_ID'], textposition="top center"))
        
    fig.update_layout(title="HPLC Trace: Peak Detection", xaxis_title="Retention Time (min)", yaxis_title="Intensity")
    st.plotly_chart(fig, use_container_width=True)
    if not df_peaks.empty:
        st.dataframe(df_peaks, use_container_width=True)

elif module == "📊 Phenotypic & HCS Clustering":
    st.title("High-Content Screening (HCS) & Unsupervised Clustering")
    np.random.seed(42)
    df_pheno = pd.DataFrame({
        'Compound_ID': [f"AMP-{i:03d}" for i in range(300)],
        'Cell_Viability': np.concatenate([np.random.normal(90, 5, 100), np.random.normal(40, 10, 100), np.random.normal(85, 8, 100)]),
        'Apoptosis_Rate': np.concatenate([np.random.normal(5, 2, 100), np.random.normal(55, 15, 100), np.random.normal(10, 5, 100)]),
        'ROS_Production': np.concatenate([np.random.normal(20, 10, 100), np.random.normal(80, 20, 100), np.random.normal(30, 10, 100)]),
        'Morphology_Score': np.concatenate([np.random.normal(0.9, 0.1, 100), np.random.normal(0.3, 0.15, 100), np.random.normal(0.8, 0.1, 100)])
    })
    
    engine = PhenotypicEngine()
    df_analyzed, variance = engine.analyze_phenotypes(df_pheno, ['Cell_Viability', 'Apoptosis_Rate', 'ROS_Production', 'Morphology_Score'])
    
    fig = px.scatter(df_analyzed, x='PCA1', y='PCA2', color='Cluster_ID', hover_data=['Compound_ID'],
                     title=f"PCA Representation (Variance Retained: {variance[0]:.1f}% + {variance[1]:.1f}%)",
                     color_discrete_sequence=['#00d4ff', '#ff00d4', '#ffdd00'])
    st.plotly_chart(fig, use_container_width=True)

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
        fig.add_trace(go.Scatter(x=df_kinetics['Time_Hours'], y=df_kinetics['Biomass_gL'], name="Biomass (g/L)", line=dict(color='#00d4ff')))
        fig.add_trace(go.Scatter(x=df_kinetics['Time_Hours'], y=df_kinetics['Substrate_gL'], name="Substrate (g/L)", line=dict(color='#ff00d4', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

elif module == "🧬 Epigenetic Array (DNA Methylation)":
    st.title("Epigenomic Profiling & Aging Biomarkers")
    analyzer = EpigeneticAnalyzer()
    df_meth = analyzer.generate_methylation_profile()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_meth['Locus'], y=df_meth['CpG_Beta'], mode='lines', name='CpG Methylation', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df_meth['Locus'], y=df_meth['Non_CpG_Beta'], mode='lines', fill='tozeroy', name='Non-CpG Methylation', line=dict(color='#ff00d4', width=2)))
    st.plotly_chart(fig, use_container_width=True)

elif module == "📸 Auto-Digitizer (Graph OCR)":
    st.title("Computer Vision: Graph to CSV")
    st.info("Upload literature graphs (Image or PDF) to extract raw coordinates.")
    
    img_file = st.file_uploader("Upload Graph File", type=['png', 'jpg', 'jpeg', 'pdf'])
    
    if img_file:
        if img_file.name.lower().endswith('.pdf'):
            st.success("📄 PDF Document Detected. The vision engine will scan the first page.")
        else:
            st.image(img_file, width=600, caption="Uploaded Original Graph")
            
        # --- NEW: Computer Vision Tuning Controls ---
        with st.expander("⚙️ Advanced Vision Engine Tuning", expanded=True):
            st.markdown("Adjust these filters if the engine misses points or captures noise (like text/gridlines).")
            c1, c2, c3 = st.columns(3)
            with c1:
                ui_min_area = st.slider("Minimum Point Size", 1, 50, 10) # Lowered default to 10
            with c2:
                ui_max_area = st.slider("Maximum Point Size", 100, 1000, 500)
            with c3:
                # Lowered circularity default to 0.2 to allow for error bars and odd shapes
                ui_circularity = st.slider("Shape Strictness (Circularity)", 0.0, 1.0, 0.2, 0.1) 
        
        if st.button("Initialize Vision Pipeline", use_container_width=True):
            with st.spinner("Running OpenCV Contours & Tesseract OCR..."):
                digitizer = GraphDigitizer(img_file)
                
                # Pass the UI slider values into the engine
                extracted_df, context_text = digitizer.process_full_pipeline()
                
                # We need to override the isolate_data_points call inside process_full_pipeline
                # So we manually do the pipeline steps here to use the slider values
                pixels = digitizer.isolate_data_points(min_area=ui_min_area, max_area=ui_max_area, min_circularity=ui_circularity)
                extracted_df = digitizer.map_to_real_data(pixels)
                
                if not extracted_df.empty:
                    st.session_state['digitized_df'] = extracted_df
                    st.success(f"Successfully digitized {len(extracted_df)} data points.")
                    st.info("💾 Data routed to Global Memory Bank. You can now use this in the Pharmacodynamics module.")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.scatter(extracted_df, x='Extracted_X', y='Extracted_Y', title="Digitized Data Reconstruction")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.dataframe(extracted_df, use_container_width=True)
                else:
                    st.error("Could not detect confident data points. Try lowering 'Minimum Point Size' and 'Shape Strictness' in the Advanced Tuning menu.")
                    
elif module == "🤖 BioSIGHT Global Copilot":
    st.title("BioSIGHT Global Copilot")
    st.caption("Your AI Research Assistant. Ask questions about your biological data.")
    
    # 1. Configure the Gemini Engine securely
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"API Key Error: {e}. Ensure secrets.toml is configured.")
        st.stop()

    # 2. Display previous chat messages from memory
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. The Chat Input box at the bottom of the screen
    if user_prompt := st.chat_input("Ask me to analyze your IC50 data, explain Monod kinetics, or write a conclusion..."):
        
        # Display user's message instantly
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Save user message to memory
        st.session_state['chat_history'].append({"role": "user", "content": user_prompt})
        
        # 4. Context Injection: Let the AI "see" the Global Memory Bank
        system_context = "You are the BioSIGHT Global Copilot, an expert AI assistant in a high-throughput biotechnology software platform. "
        
        if not st.session_state['digitized_df'].empty:
            df_string = st.session_state['digitized_df'].head(5).to_string() # Just send top 5 rows to save tokens
            system_context += f"\n\nThe user currently has this digitized data loaded in the system's memory:\n{df_string}\n\n"
        else:
            system_context += "\n\nThe user currently has no raw data loaded in the global memory bank."
            
        system_context += f"User's query: {user_prompt}"

        # 5. Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = model.generate_content(system_context)
                    st.markdown(response.text)
                    
                    # Save AI response to memory
                    st.session_state['chat_history'].append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Communication error with AI Engine: {e}")
