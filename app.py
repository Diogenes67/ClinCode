import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os

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
    
    # Text input area
    clinical_note = st.text_area(
        "Enter the clinical note for diagnosis coding:",
        height=300,
        placeholder="Enter patient symptoms, examination findings, and clinical observations here..."
    )
    
    # Classification button
    if st.button("🔍 Classify Note", type="primary", use_container_width=True):
        if clinical_note.strip():
            st.session_state.processing = True
            
            # Show processing indicator
            with st.spinner("Analyzing clinical note..."):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Mock classification logic (replace with your actual model)
                # This is where you'd integrate your actual classification model
                mock_diagnoses = [
                    {"code": "J06.9", "description": "Acute upper respiratory infection, unspecified", "confidence": 0.89},
                    {"code": "R05", "description": "Cough", "confidence": 0.76},
                    {"code": "R50.9", "description": "Fever, unspecified", "confidence": 0.65}
                ]
                
                st.session_state.classification_result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note_length": len(clinical_note),
                    "diagnoses": mock_diagnoses
                }
                
            st.session_state.processing = False
            st.success("✅ Classification completed successfully!")
        else:
            st.error("⚠️ Please enter a clinical note before classification.")

with col2:
    st.markdown("### Quick Stats")
    
    # Display statistics
    st.markdown("""
    <div class="info-card">
        <h4>📊 System Status</h4>
        <p><strong>Status:</strong> <span style="color: #28a745;">● Online</span></p>
        <p><strong>Model:</strong> ClinCode v1.0</p>
        <p><strong>Last Updated:</strong> Today</p>
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
