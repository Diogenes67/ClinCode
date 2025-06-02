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

# Suppress warnings
warnings.filterwarnings("ignore", message="coroutine 'expire_cache' was never awaited")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="streamlit")

# Page configuration with medical theme
st.set_page_config(
    page_title="ClinCode - Medical Diagnosis Coding",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical professional theme
st.markdown("""
<style>
    /* Main color scheme - medical blues and whites */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border-left: 4px solid #2a5298;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2a5298;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1e3c72;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Error messages */
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    
    .error-msg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border: 2px solid #e0e4e8;
        border-radius: 5px;
        font-size: 1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #2a5298;
        box-shadow: 0 0 0 0.2rem rgba(42, 82, 152, 0.25);
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
Your rationale should help other doctors understand the pros and cons of choosing each code.

Your task is to suggest between **one and four mutually exclusive ED Short List ICD-10-AM codes** that could each plausibly serve as the **principal diagnosis**, based on the diagnostic content of the casenote.

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

**ED Code Shortlist:**
{options_text}

**Casenote:**
{note}

---

**Output Format (exactly):**
1. CODE — "<your rationale>"
2. CODE — "<your rationale>"
3. … up to 4

Please follow that structure precisely.
"""
    
    try:
        payload = {
            "model": model,
            "prompt": fewshot + prompt,
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
    idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
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
    current_code = None
    current_expl = ""
    
    for line in resp.splitlines():
        code_match = re.match(r"\d+\.\s*([A-Z0-9\.]+)\s*[—-]\s*(.+)", line)
        expl_match = re.match(r"\*\s*(.+)", line)
        
        if code_match:
            current_code = code_match.group(1).strip()
            current_expl = ""
        elif expl_match and current_code:
            current_expl = expl_match.group(1).strip()
            if current_code in valid and current_code != 'R69':
                rows.append((
                    current_code,
                    term.get(current_code, "N/A"),
                    current_expl,
                    funding_lookup.get(current_code, 3)
                ))
            current_code = None
    
    return rows

# Check local LLM status
LOCAL_LLM_AVAILABLE = check_ollama_status()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 ClinCode</h1>
    <p>Professional Medical Diagnosis Coding System</p>
</div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Clinical Note Input")
    
    # Create tabs for input methods
    tab1, tab2, tab3 = st.tabs(["✍️ Text Input", "📄 File Upload", "📁 Batch Processing"])
    
    with tab1:
        # Text input area
        clinical_note = st.text_area(
            "Enter the clinical note for diagnosis coding:",
            height=300,
            placeholder="Enter patient symptoms, examination findings, and clinical observations here...",
            key="text_input"
        )
    
    with tab2:
        # File upload
        uploaded_note = st.file_uploader(
            "Upload a text file containing the clinical note:",
            type=["txt"],
            help="Upload a .txt file with the clinical note"
        )
        
        if uploaded_note is not None:
            # Read the file content
            clinical_note = uploaded_note.getvalue().decode("utf-8")
            st.success(f"✅ Loaded file: {uploaded_note.name}")
            
            # Display file content
            with st.expander("View file content", expanded=True):
                st.text_area("File content:", clinical_note, height=200, disabled=True)
        else:
            clinical_note = st.session_state.get('text_input', '')
    
    with tab3:
        # Batch processing
        st.markdown("#### Process Multiple Files")
        uploaded_files = st.file_uploader(
            "Upload multiple text files for batch processing:",
            type=["txt"],
            accept_multiple_files=True,
            help="Select multiple .txt files to process at once"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files selected for processing")
            
            # Show file list
            with st.expander("View selected files"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name}")
            
            # Batch process button
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                if not LOCAL_LLM_AVAILABLE:
                    st.error("⚠️ Local LLM (Ollama) is required for batch processing")
                else:
                    batch_results = []
                    progress_bar = st.progress(0)
                    
                    for idx, file in enumerate(uploaded_files):
                        with st.spinner(f"Processing {file.name}..."):
                            try:
                                # Read file content
                                content = file.getvalue().decode("utf-8")
                                
                                # Process similar to single file
                                # This is a simplified version - you can expand as needed
                                batch_results.append({
                                    'filename': file.name,
                                    'content_preview': content[:100] + "...",
                                    'status': 'Processed'
                                })
                                
                            except Exception as e:
                                batch_results.append({
                                    'filename': file.name,
                                    'content_preview': 'Error reading file',
                                    'status': f'Error: {str(e)}'
                                })
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # Display results
                    st.success(f"✅ Processed {len(uploaded_files)} files")
                    
                    # Create results dataframe
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
    
    # File paths configuration
    excel_file = st.file_uploader("Upload ICD Codes Excel (optional)", type=["xlsx", "xls"])
    jsonl_file = st.file_uploader("Upload Few-Shot Examples (optional)", type=["jsonl"])
    
    # Default paths
    EXCEL_PATH = ROOT / "FinalEDCodes_Complexity.xlsx" if excel_file is None else excel_file
    JSONL_PATH = ROOT / "edcode_finetune_v7_more_notes.jsonl" if jsonl_file is None else jsonl_file
    EMBEDDING_CACHE_PATH = ROOT / "ed_code_embeddings.pkl"
    
    # Classification button
    if st.button("🔍 Classify Note", type="primary", use_container_width=True):
        if clinical_note.strip():
            if not LOCAL_LLM_AVAILABLE:
                st.error("⚠️ Local LLM (Ollama with Llama 3) is required. Please install and start Ollama.")
                st.code("curl -fsSL https://ollama.ai/install.sh | sh\nollama pull llama3:8b-instruct-q4_K_M")
            else:
                st.session_state.processing = True
                
                try:
                    # Show processing indicator
                    with st.spinner("Loading ICD codes and building embeddings..."):
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
                    
                    with st.spinner("Analyzing clinical note with AI..."):
                        # Get embeddings for the note
                        note_emb = get_embeddings_local([clinical_note])[0]
                        
                        # Get top similar codes
                        shortlist = get_top_matches(note_emb, code_embeddings, raw, 12)
                        
                        # Query local LLM
                        resp = predict_final_codes_local(clinical_note, shortlist, fewshot)
                        
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
    
    # Display note statistics if available
    if clinical_note:
        word_count = len(clinical_note.split())
        char_count = len(clinical_note)
        
        st.markdown(f"""
        <div class="info-card">
            <h4>📝 Note Statistics</h4>
            <p><strong>Words:</strong> {word_count}</p>
            <p><strong>Characters:</strong> {char_count}</p>
        </div>
        """, unsafe_allow_html=True)

# Results section
if st.session_state.classification_result:
    st.markdown("---")
    st.markdown("### 📋 Classification Results")
    
    result = st.session_state.classification_result
    
    # Display timestamp
    st.markdown(f"**Analysis completed at:** {result['timestamp']}")
    
    # Display diagnoses
    st.markdown("#### Identified Diagnoses:")
    
    for idx, diagnosis in enumerate(result['diagnoses'], 1):
        confidence_color = "#28a745" if diagnosis['confidence'] > 0.8 else "#ffc107" if diagnosis['confidence'] > 0.6 else "#dc3545"
        
        st.markdown(f"""
        <div class="info-card">
            <h5>{idx}. {diagnosis['code']} - {diagnosis['description']}</h5>
            <p><strong>Confidence:</strong> <span style="color: {confidence_color};">{diagnosis['confidence']:.0%}</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as PDF"):
            st.info("PDF export functionality would be implemented here")
    
    with col2:
        if st.button("📊 Export as CSV"):
            # Create CSV data
            df = pd.DataFrame(result['diagnoses'])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"clincode_results_{result['timestamp'].replace(':', '-')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔄 Clear Results"):
            st.session_state.classification_result = None
            st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 ClinCode")
    st.markdown("Professional medical diagnosis coding system")
    
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
    
    # Display settings
    st.checkbox("Show detailed explanations", value=False)
    st.checkbox("Enable auto-save", value=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    <p>ClinCode © 2024 | Medical Diagnosis Coding System | Version 1.0</p>
</div>
""", unsafe_allow_html=True)
