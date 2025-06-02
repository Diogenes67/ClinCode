import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os
import pickle
import re
from numpy.linalg import norm
import plotly.graph_objects as go
import requests
import warnings
from pathlib import Path
import docx
import PyPDF2
import io

# Suppress warnings
warnings.filterwarnings("ignore", message="coroutine 'expire_cache' was never awaited")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="streamlit")

# Page configuration with medical theme
st.set_page_config(
    page_title="ClinCode - Medical Diagnosis Coding",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar closed by default
)

# Custom CSS for Amplar Health theme
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main color scheme - Amplar Health colors */
    .stApp {
        background-color: #f9fafb;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hero gradient background */
    .hero-gradient {
        background: linear-gradient(135deg, #e6f2ff 0%, #f0f9ff 50%, #ffffff 100%);
        padding: 3rem 0;
        margin: -3rem -3rem 2rem -3rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Header styling */
    .main-header {
        background: transparent;
        color: #5a6069;
        padding: 0 2rem;
        margin-bottom: 0;
        text-align: left;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3.5rem;
        font-weight: 700;
        color: #4a90e2;
        line-height: 1.1;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .medical-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.25rem;
        color: #6b7280;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Card styling with shadows */
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }
    
    .info-card {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Button styling - Amplar blue */
    .stButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%) !important;
        color: white !important;
        border: none;
        padding: 0.875rem 2.5rem;
        font-size: 1.125rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.01em;
        box-shadow: 0 4px 6px rgba(74, 144, 226, 0.2);
    }
    
    /* Force white text on all button states */
    .stButton > button span {
        color: white !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #357abd 0%, #2968a3 100%) !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(74, 144, 226, 0.3);
    }
    
    .stButton > button:hover span {
        color: white !important;
    }
    
    .stButton > button:active,
    .stButton > button:focus {
        color: white !important;
    }
    
    .stButton > button:active span,
    .stButton > button:focus span {
        color: white !important;
    }
    
    /* Tab styling with icons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #6b7280;
        border: none;
        padding: 1rem 0;
        font-weight: 500;
        font-size: 1rem;
        position: relative;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #4a90e2;
    }
    
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4a90e2 0%, #357abd 100%);
        border-radius: 3px 3px 0 0;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        padding: 1rem;
        transition: all 0.2s ease;
        background-color: #fcfcfc;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
        background-color: white;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #f9fafb 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background: linear-gradient(135deg, #e6f2ff 0%, #f0f9ff 100%);
        border-color: #4a90e2;
        transform: scale(1.01);
    }
    
    /* Progress steps styling */
    .progress-step {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .progress-step.active {
        background: linear-gradient(135deg, #e6f2ff 0%, #f0f9ff 100%);
        border: 1px solid #4a90e2;
    }
    
    .progress-step-icon {
        width: 40px;
        height: 40px;
        background: #e5e7eb;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-weight: 600;
        color: #6b7280;
    }
    
    .progress-step.active .progress-step-icon {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #5a6069;
        font-weight: 600;
    }
    
    p {
        color: #6b7280;
        line-height: 1.6;
    }
    
    /* Links */
    a {
        color: #4a90e2;
        text-decoration: none;
    }
    
    a:hover {
        color: #357abd;
        text-decoration: underline;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'show_explanations' not in st.session_state:
    st.session_state.show_explanations = True

# Set base path
ROOT = Path(__file__).parent.resolve()

# === Ollama Integration Functions ===
def check_ollama_status():
    """Check if Ollama is running and Llama model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            return 'llama3:8b-instruct-q4_K_M' in available_models
        return False
    except:
        return False

def predict_final_codes_local(note, shortlist_df, fewshot, model="llama3:8b-instruct-q4_K_M"):
    """Local LLM processing using Ollama."""
    options_text = "\n".join(
        f"{r['ED Short List code']} — {r['ED Short List Term']}" for _, r in shortlist_df.iterrows()
    )
    
    prompt = f"""You are the head of the emergency department and an expert clinical coder.

### Examples of coding (for reference only):
{fewshot}

### NOW, analyze THIS NEW case:

Your task is to suggest between **one and four mutually exclusive ED Short List ICD-10-AM codes** that could each plausibly serve as the **principal diagnosis**, based on the diagnostic content of the casenote below.

These codes are **not** intended to build a combined clinical picture — rather, they should reflect **alternative coding options**, depending on the coder's interpretation and emphasis. **Each code must stand on its own** as a valid representation of the case presentation.

---

**How to think it through (show your work):**
1. Identify the single finding or cluster of findings that most tightly matches one code.
2. Pick that code as **#1 (best fit)** and provide a clear justification:
   - Show exactly which language in the note drives your choice
   - Highlight why it's more specific or higher-priority than the next option
3. Repeat for up to **4 total**, each time choosing the next-best fit.
4. If no highly specific match remains, choose the least-specific fallback — but **do not** use R69.
5. **Do not** list comorbidities or incidental findings unless they truly dominate the presentation.

---

**ED Code Shortlist for THIS case:**
{options_text}

**Current Casenote to analyze:**
{note}

---

**Output Format (exactly):**
1. CODE — "<your rationale>"
2. CODE — "<your rationale>"
3. … up to 4

Please analyze ONLY the current casenote above and provide your coding suggestions.
"""
    
    try:
        payload = {
            "model": model,
            "prompt": prompt,  # Remove fewshot from being prepended
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1000}
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            st.error(f"Local LLM error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error calling local LLM: {e}")
        return None

# === File Processing Functions ===
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    """Process uploaded file based on its type."""
    if uploaded_file is None:
        return None
    
    file_type = uploaded_file.type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'txt' or file_type == 'text/plain':
            # Handle text files
            return uploaded_file.getvalue().decode("utf-8")
        
        elif file_extension == 'pdf' or file_type == 'application/pdf':
            # Handle PDF files
            return extract_text_from_pdf(uploaded_file)
        
        elif file_extension in ['docx', 'doc'] or file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            # Handle Word documents
            return extract_text_from_docx(uploaded_file)
        
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# === Utility Functions ===
def cosine(u, v):
    """Return cosine similarity between vectors u and v."""
    return np.dot(u, v) / (norm(u) * norm(v))

@st.cache_data
def get_embeddings_local(texts):
    """Obtain embeddings using local SentenceTransformer model."""
    texts = list(texts)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
        embeddings = model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
    except ImportError:
        st.error("sentence-transformers not installed. Please install it with: pip install sentence-transformers")
        return None
    except Exception as e:
        st.error(f"Error with local embeddings: {e}")
        return None

@st.cache_data
def build_code_embeddings(descriptions, cache_path, use_local=True):
    """Build or load cached embeddings for the code descriptions."""
    cache_path = Path(cache_path)
    
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                embeds = pickle.load(f)
            return embeds
        except Exception as e:
            st.sidebar.warning(f"Error loading cache: {e}")
    
    # Generate embeddings
    with st.sidebar.status("Generating local embeddings..."):
        embeds = get_embeddings_local(descriptions)
        
        if embeds is not None and len(embeds) > 0:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(embeds, f)
                st.sidebar.success(f"Generated and cached {len(embeds)} embeddings")
            except Exception as e:
                st.sidebar.warning(f"Failed to cache embeddings: {e}")
        else:
            st.sidebar.error("Failed to generate embeddings")
            return None
    
    return embeds

@st.cache_data
def get_top_matches(note_emb, code_embs, df, top_n=5):
    """Compute cosine similarity between note embedding and each code embedding."""
    sims = [cosine(note_emb, e) for e in code_embs]
    
    # Get indices sorted by similarity
    idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    
    # For better anatomical matching, filter results
    # If the note mentions specific body parts, deprioritize codes for other body parts
    note_lower = st.session_state.get('note_text', '').lower()
    
    # Common anatomical terms to check
    anatomical_filters = {
        'knee': ['ankle', 'foot', 'hip', 'shoulder', 'elbow', 'wrist', 'spine'],
        'ankle': ['knee', 'hip', 'shoulder', 'elbow', 'wrist', 'spine'],
        'shoulder': ['knee', 'ankle', 'hip', 'elbow', 'wrist', 'spine'],
        'back': ['knee', 'ankle', 'shoulder', 'elbow', 'wrist'],
        'spine': ['knee', 'ankle', 'shoulder', 'elbow', 'wrist'],
    }
    
    # Check which body part is mentioned
    mentioned_parts = []
    for part in anatomical_filters.keys():
        if part in note_lower:
            mentioned_parts.append(part)
    
    # Filter indices if specific body parts are mentioned
    if mentioned_parts:
        filtered_idx = []
        excluded_terms = set()
        for part in mentioned_parts:
            excluded_terms.update(anatomical_filters.get(part, []))
        
        for i in idx:
            term_lower = df.iloc[i]['ED Short List Term'].lower()
            # Check if this code is for a different body part
            exclude = False
            for excluded in excluded_terms:
                if excluded in term_lower:
                    exclude = True
                    break
            
            if not exclude:
                filtered_idx.append(i)
            
            if len(filtered_idx) >= top_n:
                break
        
        # If we filtered too much, add some back
        if len(filtered_idx) < top_n:
            for i in idx:
                if i not in filtered_idx:
                    filtered_idx.append(i)
                if len(filtered_idx) >= top_n:
                    break
        
        idx = filtered_idx[:top_n]
    else:
        idx = idx[:top_n]
    
    top = df.iloc[idx].copy()
    top['Similarity'] = [sims[i] for i in idx]
    return top

@st.cache_data
def load_examples(path, limit=3):
    """Load few-shot examples from a .jsonl file for prompt context."""
    path = Path(path)
    if not path.exists():
        st.error(f"Example file not found: {path}")
        return ""
    
    ex = []
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= limit: break
                d = json.loads(line)
                ex.append(
                    f"Casenote:\n{d['messages'][0]['content']}\nAnswer:\n{d['messages'][1]['content']}"
                )
        return "\n\n---\n\n".join(ex) + "\n\n---\n\n"
    except Exception as e:
        st.error(f"Error loading examples: {e}")
        return ""

def parse_response(resp, df):
    """Parse the LLM response to extract ICD codes and explanations."""
    valid = set(df['ED Short List code'].astype(str).str.strip())
    term = dict(zip(df['ED Short List code'], df['ED Short List Term']))
    funding_lookup = dict(zip(df['ED Short List code'], df['Scale'].fillna(3).astype(int)))
    
    rows = []
    lines = resp.splitlines()
    i = 0
    
    # Skip any example content if present
    while i < len(lines):
        line = lines[i].strip()
        # Skip lines that look like casenote headers from examples
        if line.startswith("**Casenote:**") or line.startswith("Casenote:"):
            # Skip until we find the actual current case
            while i < len(lines) and not "Current Casenote" in lines[i]:
                i += 1
        i += 1
    
    # Reset to start and parse the actual response
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for pattern: "1. **CODE** — Term" or "1. CODE — Term"
        patterns = [
            r"\d+\.\s*\*\*([A-Z0-9\.]+)\*\*\s*[—–-]\s*(.*)",  # 1. **CODE** — Term
            r"\d+\.\s*([A-Z0-9\.]+)\s*[—–-]\s*(.*)",          # 1. CODE — Term
        ]
        
        matched = False
        for pattern in patterns:
            m = re.match(pattern, line)
            if m:
                code = m.group(1).strip()
                # The term after the dash might not match exactly, so we'll use our lookup
                
                # Now look for the explanation in the next lines (bullet points)
                explanation_lines = []
                i += 1
                
                # Collect all bullet point lines that follow
                while i < len(lines) and (lines[i].strip().startswith('*') or lines[i].strip().startswith('-') or (lines[i].strip() and not re.match(r'\d+\.', lines[i].strip()))):
                    exp_line = lines[i].strip()
                    # Remove bullet point markers
                    exp_line = re.sub(r'^[\*\-]\s*', '', exp_line)
                    if exp_line:
                        explanation_lines.append(exp_line)
                    i += 1
                
                explanation = ' '.join(explanation_lines)
                
                if code in valid and code != 'R69':
                    rows.append((
                        code,
                        term.get(code, "N/A"),  # Use our term lookup
                        explanation,
                        funding_lookup.get(code, 3)
                    ))
                    matched = True
                    break
        
        if not matched:
            i += 1
    
    return rows

# Check local LLM status
LOCAL_LLM_AVAILABLE = check_ollama_status()

# Header with hero gradient
st.markdown("""
<div class="hero-gradient">
    <div class="main-header">
        <h1>
            <span class="medical-icon">⚕️</span>
            ClinCode
        </h1>
        <p>Professional Medical Diagnosis Coding System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Set file paths based on sidebar uploads (defined early to be available everywhere)
EXCEL_PATH = ROOT / "FinalEDCodes_Complexity.xlsx"
JSONL_PATH = ROOT / "edcode_finetune_v7_more_notes.jsonl"
EMBEDDING_CACHE_PATH = ROOT / "ed_code_embeddings.pkl"

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Clinical Note Input")
    
    # Create tabs for input methods with icons
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📄 File Upload", "📁 Batch Processing"])
    
    with tab1:
        # Text input area
        clinical_note = st.text_area(
            "Enter the clinical note for diagnosis coding:",
            height=300,
            placeholder="Enter patient symptoms, examination findings, and clinical observations here..."
        )
    
    with tab1:
        # Text input area
        clinical_note = st.text_area(
            "Enter the clinical note for diagnosis coding:",
            height=300,
            placeholder="Enter patient symptoms, examination findings, and clinical observations here...",
            key="text_input"
        )
    
    with tab2:
        # File upload with drag-and-drop
        st.markdown("#### 📄 Upload Clinical Note")
        st.markdown("**Drag and drop** or **click to browse** - Supports TXT, PDF, and DOCX files")
        
        uploaded_note = st.file_uploader(
            "Choose a file",
            type=["txt", "pdf", "docx", "doc"],
            help="Upload a file containing the clinical note. Supported formats: TXT, PDF, DOCX",
            key="single_file_upload"
        )
        
        if uploaded_note is not None:
            # Extract text based on file type
            with st.spinner(f"Processing {uploaded_note.name}..."):
                extracted_text = process_uploaded_file(uploaded_note)
            
            if extracted_text:
                clinical_note = extracted_text
                st.success(f"✅ Successfully loaded: {uploaded_note.name}")
                
                # Display file content
                with st.expander("View extracted content", expanded=True):
                    st.text_area("Content:", clinical_note, height=200, disabled=True)
            else:
                clinical_note = ""
                st.error("Failed to extract text from file")
        else:
            clinical_note = ""
    
    with tab3:
        # Batch processing with multiple file formats
        st.markdown("#### 📁 Batch Process Multiple Files")
        st.markdown("**Drag and drop multiple files** or **click to browse**")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["txt", "pdf", "docx", "doc"],
            accept_multiple_files=True,
            help="Select multiple files to process at once. Supported: TXT, PDF, DOCX",
            key="batch_file_upload"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files ready for processing")
            
            # Show file list with types
            with st.expander("View selected files", expanded=True):
                file_data = []
                for file in uploaded_files:
                    file_data.append({
                        "File Name": file.name,
                        "Type": file.name.split('.')[-1].upper(),
                        "Size (KB)": f"{len(file.getvalue()) / 1024:.1f}"
                    })
                st.dataframe(pd.DataFrame(file_data), use_container_width=True)
            
            # Batch process button
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                if not LOCAL_LLM_AVAILABLE:
                    st.error("⚠️ Local LLM (Ollama) is required for batch processing")
                else:
                    batch_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")
                        
                        try:
                            # Extract content based on file type
                            content = process_uploaded_file(file)
                            
                            if content:
                                # Here you would add the actual classification logic
                                # For now, just showing successful extraction
                                batch_results.append({
                                    'filename': file.name,
                                    'file_type': file.name.split('.')[-1].upper(),
                                    'content_preview': content[:100] + "..." if len(content) > 100 else content,
                                    'word_count': len(content.split()),
                                    'status': '✅ Processed'
                                })
                            else:
                                batch_results.append({
                                    'filename': file.name,
                                    'file_type': file.name.split('.')[-1].upper(),
                                    'content_preview': 'Failed to extract',
                                    'word_count': 0,
                                    'status': '❌ Error'
                                })
                                
                        except Exception as e:
                            batch_results.append({
                                'filename': file.name,
                                'file_type': file.name.split('.')[-1].upper(),
                                'content_preview': 'Error',
                                'word_count': 0,
                                'status': f'❌ {str(e)}'
                            })
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    status_text.empty()
                    st.success(f"✅ Completed processing {len(uploaded_files)} files")
                    
                    # Display results
                    results_df = pd.DataFrame(batch_results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"clincode_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    clinical_note = None  # Clear for batch mode
    
    st.markdown("---")
    
    # Classification button
    if st.button("🔍 Classify Note", type="primary", use_container_width=True, help="Analyze the clinical note to suggest ICD-10 codes"):
        if clinical_note.strip():
            if not LOCAL_LLM_AVAILABLE:
                st.error("⚠️ Local LLM (Ollama with Llama 3) is required. Please install and start Ollama.")
                st.code("curl -fsSL https://ollama.ai/install.sh | sh\nollama pull llama3:8b-instruct-q4_K_M")
            else:
                st.session_state.processing = True
                
                try:
                    # Create progress container
                    progress_container = st.container()
                    
                    with progress_container:
                        # Step 1: Loading data
                        st.markdown("""
                        <div class="progress-step active">
                            <div class="progress-step-icon">1</div>
                            <div>
                                <strong>Loading ICD codes and embeddings</strong><br>
                                <small>Preparing classification system...</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Load Excel data
                        if 'excel_file' in st.session_state and st.session_state.excel_file is not None:
                            EXCEL_PATH = st.session_state.excel_file
                        if 'jsonl_file' in st.session_state and st.session_state.jsonl_file is not None:
                            JSONL_PATH = st.session_state.jsonl_file
                        
                        # Load Excel data
                        if isinstance(EXCEL_PATH, Path):
                            raw = pd.read_excel(EXCEL_PATH)
                        else:
                            raw = pd.read_excel(EXCEL_PATH)
                        
                        raw.columns = raw.columns.str.strip()
                        raw = raw.rename(columns={
                            "ED Short": "ED Short List code",
                            "Diagnosis": "ED Short List Term",
                            "Descriptor": "ED Short List Included conditions"
                        })
                        desc_list = (raw["ED Short List Term"] + ". " + raw["ED Short List Included conditions"].fillna(""))
                        
                        # Build embeddings
                        code_embeddings = build_code_embeddings(desc_list, EMBEDDING_CACHE_PATH, use_local=True)
                        
                        # Load few-shot examples
                        if isinstance(JSONL_PATH, Path):
                            fewshot = load_examples(JSONL_PATH)
                        else:
                            fewshot = ""
                    
                    # Step 2: Analyzing text
                    with progress_container:
                        st.markdown("""
                        <div class="progress-step active">
                            <div class="progress-step-icon">2</div>
                            <div>
                                <strong>Analyzing clinical note</strong><br>
                                <small>Finding relevant diagnosis codes...</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get embeddings for the note
                        note_emb = get_embeddings_local([clinical_note])[0]
                        
                        # Get top similar codes
                        shortlist = get_top_matches(note_emb, code_embeddings, raw, 12)
                    
                    # Step 3: Generating results
                    with progress_container:
                        st.markdown("""
                        <div class="progress-step active">
                            <div class="progress-step-icon">3</div>
                            <div>
                                <strong>Generating diagnosis suggestions</strong><br>
                                <small>AI analyzing best matches...</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Query local LLM
                        resp = predict_final_codes_local(clinical_note, shortlist, fewshot)
                        
                        raw.columns = raw.columns.str.strip()
                        raw = raw.rename(columns={
                            "ED Short": "ED Short List code",
                            "Diagnosis": "ED Short List Term",
                            "Descriptor": "ED Short List Included conditions"
                        })
                        desc_list = (raw["ED Short List Term"] + ". " + raw["ED Short List Included conditions"].fillna(""))
                        
                        # Build embeddings
                        code_embeddings = build_code_embeddings(desc_list, EMBEDDING_CACHE_PATH, use_local=True)
                        
                        # Load few-shot examples
                        if isinstance(JSONL_PATH, Path):
                            fewshot = load_examples(JSONL_PATH)
                        else:
                            fewshot = ""
                    
                    with st.spinner("Analyzing clinical note with AI..."):
                        # Get embeddings for the note
                        note_emb = get_embeddings_local([clinical_note])[0]
                        
                        # Get top similar codes
                        shortlist = get_top_matches(note_emb, code_embeddings, raw, 12)
                        
                        # Query local LLM
                        resp = predict_final_codes_local(clinical_note, shortlist, fewshot)
                        
                        if resp:
                            # Debug: Show raw response in expander
                            with st.expander("🔍 Debug: Raw LLM Response"):
                                st.code(resp)
                            
                            # Parse LLM response
                            parsed = parse_response(resp, raw)
                            
                            # Debug: Show parsed results
                            with st.expander("🔍 Debug: Parsed Results"):
                                st.write(f"Number of codes found: {len(parsed)}")
                                for item in parsed:
                                    st.write(item)
                        
                        if resp:
                            # Parse LLM response
                            parsed = parse_response(resp, raw)
                            
                            # Format results for display
                            diagnoses = []
                            for code, term, explanation, complexity in parsed:
                                confidence = 0.95 - (len(diagnoses) * 0.1)  # Simulate decreasing confidence
                                diagnoses.append({
                                    "code": code,
                                    "description": term,
                                    "confidence": confidence,
                                    "explanation": explanation,
                                    "complexity": complexity
                                })
                            
                            st.session_state.classification_result = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "note_length": len(clinical_note),
                                "diagnoses": diagnoses,
                                "shortlist": shortlist,
                                "raw_response": resp
                            }
                            
                            st.session_state.processing = False
                            st.success("✅ Classification completed successfully!")
                        else:
                            st.error("Failed to get response from AI model.")
                            st.session_state.processing = False
                        
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
                    st.session_state.processing = False
        else:
            st.error("⚠️ Please enter a clinical note before classification.")

with col2:
    st.markdown("### Quick Stats")
    
    # Display LLM status
    if LOCAL_LLM_AVAILABLE:
        llm_status = '<span style="color: #28a745;">● Online</span>'
        llm_text = "Local AI Ready"
    else:
        llm_status = '<span style="color: #dc3545;">● Offline</span>'
        llm_text = "Ollama Required"
    
    # Display statistics
    st.markdown(f"""
    <div class="info-card">
        <h4>📊 System Status</h4>
        <p><strong>AI Status:</strong> {llm_status}</p>
        <p><strong>Model:</strong> {llm_text}</p>
        <p><strong>Privacy:</strong> 🔒 Local Processing</p>
    </div>
    """, unsafe_allow_html=True)

# Results section
if st.session_state.classification_result:
    st.markdown("---")
    st.markdown("### 📋 Classification Results")
    
    result = st.session_state.classification_result
    
    # Display timestamp
    st.markdown(f"**Analysis completed at:** {result['timestamp']}")
    
    # Create results dataframe for Plotly table
    if result['diagnoses']:
        # Prepare data for the table
        codes = []
        terms = []
        explanations = []
        complexities = []
        
        complexity_emojis = {
            1: "🟣", 2: "🔵", 3: "🟢",
            4: "🟡", 5: "🟠", 6: "🔴"
        }
        
        for diagnosis in result['diagnoses']:
            codes.append(diagnosis['code'])
            terms.append(diagnosis['description'])
            explanations.append(diagnosis.get('explanation', 'No explanation available'))
            complexity = diagnosis.get('complexity', 3)
            complexities.append(f"{complexity_emojis.get(complexity, '🟢')} {complexity}")
        
        # Create Plotly table
        fig = go.Figure(data=[go.Table(
            columnwidth=[80, 200, 500, 100],
            header=dict(
                values=["Code", "Term", "Explanation", "Complexity"],
                fill_color='rgb(74, 144, 226)',  # Amplar blue
                font=dict(color='white', size=14, family='Inter'),
                align='left',
                height=40
            ),
            cells=dict(
                values=[codes, terms, explanations, complexities],
                fill_color='rgb(255, 255, 255)',  # White background
                align='left',
                font=dict(size=13, color='rgb(90, 96, 105)', family='Inter'),  # Gray text
                height=80,  # Good height for readability
                line=dict(width=1, color='rgb(229, 231, 235)')  # Light gray borders
            )
        )])
        
        # Calculate appropriate height - ensure all rows are visible
        table_height = 40 + (80 * len(codes)) + 20  # header + (row_height * num_rows) + padding
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=table_height  # Dynamic height to show all results
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as PDF"):
                st.info("PDF export functionality would be implemented here")
        
        with col2:
            # Create CSV data
            export_df = pd.DataFrame({
                'Code': codes,
                'Term': terms,
                'Explanation': explanations,
                'Complexity': [c.split()[-1] for c in complexities]  # Extract just the number
            })
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📊 Export as CSV",
                data=csv,
                file_name=f"clincode_results_{result['timestamp'].replace(':', '-')}.csv",
                mime="text/csv"
            )
        
        with col3:
            if st.button("🔄 Clear Results"):
                st.session_state.classification_result = None
                st.rerun()
    else:
        st.warning("No valid codes extracted from the response.")
    
    # Show debug information if available
    if 'shortlist' in result:
        with st.expander("📊 Embedding Shortlist", expanded=False):
            st.dataframe(
                result['shortlist'][["ED Short List code", "ED Short List Term", "Similarity"]],
                use_container_width=True,
                hide_index=True
            )
    
    if 'raw_response' in result:
        with st.expander("🤖 LLM Raw Response", expanded=False):
            st.code(result['raw_response'])

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 ClinCode")
    st.markdown("Professional medical diagnosis coding system")
    
    st.markdown("---")
    
    # LLM Status
    if LOCAL_LLM_AVAILABLE:
        st.success("✅ Local LLM Ready")
    else:
        st.error("❌ Local LLM Offline")
        st.caption("Install Ollama to enable")
    
    st.markdown("---")
    
    st.markdown("### 📁 Data Files")
    
    # File uploaders in sidebar
    excel_file = st.file_uploader(
        "ICD Codes Excel", 
        type=["xlsx", "xls"],
        help="Upload custom ICD codes file (optional - uses default if not provided)",
        key="excel_file"
    )
    
    jsonl_file = st.file_uploader(
        "Few-Shot Examples", 
        type=["jsonl"],
        help="Upload training examples (optional - uses default if not provided)",
        key="jsonl_file"
    )
    
    # Update global paths when files are uploaded
    if excel_file is not None:
        EXCEL_PATH = excel_file
        st.caption(f"✅ Using: {excel_file.name}")
    else:
        EXCEL_PATH = ROOT / "FinalEDCodes_Complexity.xlsx"
        st.caption("📄 Using default ICD codes")
        
    if jsonl_file is not None:
        JSONL_PATH = jsonl_file
        st.caption(f"✅ Using: {jsonl_file.name}")
    else:
        JSONL_PATH = ROOT / "edcode_finetune_v7_more_notes.jsonl"
        st.caption("📄 Using default examples")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    ClinCode is an AI-powered medical diagnosis coding tool designed to assist healthcare professionals in accurately coding clinical notes.
    
    **Features:**
    - Real-time diagnosis classification
    - ICD-10 code suggestions
    - Confidence scoring
    - Export capabilities
    """)
    
    st.markdown("---")
    
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [ICD-10 Guidelines](https://www.who.int/classifications/icd/en/)
    - [Clinical Documentation](https://www.ahima.org/)
    - [Support & Help](mailto:support@clincode.com)
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    
    # Model selection
    model_type = st.selectbox(
        "Classification Model",
        ["ClinCode Standard", "ClinCode Advanced", "ClinCode Specialized"]
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05
    )
    
    # Display settings - Show detailed explanations ON by default
    st.session_state.show_explanations = st.checkbox("Show detailed explanations", value=st.session_state.show_explanations)
    auto_save = st.checkbox("Enable auto-save", value=True)

# Update file paths based on sidebar uploads
if 'excel_file' in st.session_state and st.session_state.excel_file is not None:
    EXCEL_PATH = st.session_state.excel_file
    
if 'jsonl_file' in st.session_state and st.session_state.jsonl_file is not None:
    JSONL_PATH = st.session_state.jsonl_file

# Footer
st.markdown("---")

# Complexity Scale Legend
st.markdown("""
### 🧾 Complexity Scale Legend

The **Complexity** value reflects the typical resource use associated with each diagnosis code in the Emergency Department setting, based on historical funding data.

<table style="width:100%; font-size:16px; border-collapse:collapse;">
  <thead>
    <tr>
      <th align="left">Scale</th>
      <th align="left">Funding Range (AUD)</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>🟣 1</td><td>≤ $499</td><td>Minimal complexity</td></tr>
    <tr><td>🔵 2</td><td>$500 – $699</td><td>Low complexity</td></tr>
    <tr><td>🟢 3</td><td>$700 – $899</td><td>Moderate complexity</td></tr>
    <tr><td>🟡 4</td><td>$900 – $1099</td><td>High complexity</td></tr>
    <tr><td>🟠 5</td><td>$1100 – $1449</td><td>Significant complexity</td></tr>
    <tr><td>🔴 6</td><td>≥ $1450</td><td>Very high complexity</td></tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    <p>ClinCode © 2024 | Medical Diagnosis Coding System | Version 1.0</p>
</div>
""", unsafe_allow_html=True)
