# app.py - Advanced Clinical Dermoscopy & Skin Lesion AI Classifier
# Powered by EfficientNetV2-M + EfficientNet-B4 Dual Backbone with Hierarchical Routing

import os
import time
import json
import numpy as np
import pandas as pd
import cv2
import streamlit as st

import inference as inf
import pipeline_tools as pt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------
# Page Configuration & Styling
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="DermoScan AI | Clinical Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Medical / Clinical Dark Theme CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --primary-cyan: #0ea5e9;
    --primary-glow: rgba(14, 165, 233, 0.25);
    --bg-dark: #090e17;
    --card-bg: rgba(15, 23, 42, 0.75);
    --border-color: rgba(56, 189, 248, 0.16);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --danger-red: #ef4444;
    --danger-glow: rgba(239, 68, 68, 0.25);
    --warning-amber: #f59e0b;
    --success-green: #10b981;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Background */
.stApp {
    background: radial-gradient(circle at 15% 10%, #0d192e 0%, #070b12 85%);
    color: var(--text-primary);
}

/* Hide default streamlit header clutter */
header[data-testid="stHeader"] {
    background: rgba(9, 14, 23, 0.85);
    backdrop-filter: blur(10px);
}

/* Cards & Glassmorphic Containers */
.medical-card {
    background: var(--card-bg);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.medical-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    box-shadow: 0 14px 34px -10px rgba(14, 165, 233, 0.15);
}

/* Header Banner */
.app-header {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.08) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 18px;
    padding: 24px 30px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.app-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.15) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(120deg, #ffffff 40%, #7dd3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 850px;
    line-height: 1.5;
}

/* Badges & Pills */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 8px;
    margin-bottom: 6px;
}
.badge-blue {
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(14, 165, 233, 0.35);
}
.badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.badge-red {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Diagnosis Hero Card */
.diagnosis-hero {
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    position: relative;
    border: 1px solid;
    transition: all 0.3s ease;
}
.diagnosis-hero.malignant {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(15, 23, 42, 0.85) 100%);
    border-color: rgba(239, 68, 68, 0.4);
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.15);
}
.diagnosis-hero.benign {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.85) 100%);
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
}
.diagnosis-hero.precancerous {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(15, 23, 42, 0.85) 100%);
    border-color: rgba(245, 158, 11, 0.4);
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.15);
}

.diag-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}

/* Metric Display Unit */
.metric-box {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.82rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Scanning Animation Frame */
.scanner-viewport {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid #0ea5e9;
    box-shadow: 0 0 25px rgba(14, 165, 233, 0.3);
}
.scan-laser {
    position: absolute;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #38bdf8, #0ea5e9, #38bdf8, transparent);
    box-shadow: 0 0 15px 3px rgba(56, 189, 248, 0.85);
    animation: scanAnimation 2.2s infinite alternate ease-in-out;
    z-index: 10;
}
@keyframes scanAnimation {
    0% { top: 2%; opacity: 0.8; }
    50% { opacity: 1; }
    100% { top: 96%; opacity: 0.8; }
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(56, 189, 248, 0.12);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 600;
    color: #94a3b8;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(14, 165, 233, 0.18) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(14, 165, 233, 0.35) !important;
}

/* Progress Bars */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0ea5e9 0%, #38bdf8 100%);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #080d17;
    border-right: 1px solid rgba(56, 189, 248, 0.12);
}

/* Interactive Accordion & Expanders */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.45) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* ABCDE Feature Tag */
.abcde-card {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid var(--primary-cyan);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# Model Caching & Initialization
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Dual-Trunk EfficientNet Backbone...")
def get_model():
    return inf.load_model()

try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading neural network weights: {str(e)}")
    st.stop()


# --------------------------------------------------------------------------------
# Header Banner
# --------------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;">
        <span class="badge-pill badge-blue">⚡ EfficientNetV2-M + EfficientNet-B4</span>
        <span class="badge-pill badge-green">🎯 ISIC 2019 Benchmark: 80.99% Acc</span>
        <span class="badge-pill badge-amber">🧬 Hierarchical Clinical Tree</span>
    </div>
    <div class="header-title">DermoScan AI: Intelligent Dermoscopy & Lesion Classifier</div>
    <div class="header-subtitle">
        Clinical computer vision system equipped with full DullRazor hair artifact filtering, 
        Shades-of-Gray illumination normalization, real-time lesion morphometrics (ABCDE criteria), 
        and 5-node hierarchical neural network triage.
    </div>
</div>
""", unsafe_allow_html=True)

# Regulatory Disclaimer Banner
st.info(
    "⚠️ **Investigational Research Prototype**: Trained on the public ISIC 2019 dataset. "
    "Designed for clinical decision support and educational research. "
    "Not a standalone diagnostic device. Always consult a licensed board-certified dermatologist for any skin lesion."
)


# --------------------------------------------------------------------------------
# Sidebar: Patient Metadata & Advanced Inference Controls
# --------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📋 Patient Clinical Metadata")
    st.caption("Patient characteristics are encoded and fed to the neural fusion layer.")
    
    patient_meta = {}
    if inf.USE_META:
        age_known = st.checkbox("Age known", value=True)
        if age_known:
            age = st.number_input("Patient Age (Years)", min_value=1, max_value=115, value=52, step=1)
            patient_meta['age'] = f"{age} yrs"
        else:
            age = None
            patient_meta['age'] = "Unknown"
            
        sex = st.selectbox("Biological Sex", ["male", "female", "unknown"], index=0)
        patient_meta['sex'] = sex
        
        all_sites = inf.CFG['meta']['sites']
        default_site_idx = all_sites.index('anterior torso') if 'anterior torso' in all_sites else 0
        site = st.selectbox("Anatomical Body Site", all_sites, index=default_site_idx)
        patient_meta['site'] = site
        
        encoded_meta = inf.meta_vector(age, sex, site)
    else:
        st.info("Model configuration operates in vision-only mode.")
        encoded_meta = None
        patient_meta = {'age': 'N/A', 'sex': 'N/A', 'site': 'N/A'}

    st.markdown("---")
    st.markdown("### ⚙️ Inference & Scanning Controls")
    
    routing_mode = st.selectbox(
        "Hierarchical Routing Engine",
        options=["hard", "soft", "blend", "flat"],
        index=0,
        help="Hard: Discrete clinical branch decisions. Soft: Multi-path joint probability. Blend: Ensemble of Soft + Flat head. Flat: Direct 7-class head."
    )
    
    n_tta = st.slider(
        "Test-Time Augmentation (TTA)",
        min_value=1,
        max_value=8,
        value=8,
        help="Number of geometric rotation and flip views averaged during inference. 1 = Fastest CPU inference, 8 = Maximum diagnostic stability."
    )
    
    enable_scan_sim = st.checkbox("Display Scanning Animation", value=True, help="Simulates real-time clinical laser scanning during preprocessing.")

    st.markdown("---")
    st.markdown("### 📊 Model Architecture Specs")
    st.markdown(f"""
    - **Dual Backbone**: EfficientNetV2-M + B4
    - **Input Dimensions**: 320 × 320 × 3
    - **Preproc Preset**: `{inf.CFG.get('preproc', 'light')}`
    - **Hair Removal**: Blackhat + DullRazor
    - **Illumination**: Shades of Gray ($p=6$)
    - **Held-out Acc**: `{inf.CFG.get('test_overall_accuracy', 80.99):.2f}%`
    """)


# --------------------------------------------------------------------------------
# Main Area: Input Selection (Upload or Built-in Clinical Samples)
# --------------------------------------------------------------------------------
col_input1, col_input2 = st.columns([1.4, 1.0])

selected_sample = None
with col_input1:
    uploaded_file = st.file_uploader(
        "📤 Upload Dermoscopic Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="High-magnification dermatoscope photograph or clinical macro lesion capture."
    )

with col_input2:
    st.markdown("**💡 Or Test with Clinical Demo Samples:**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        if st.button("🩸 Sample 1\n(Melanoma)", use_container_width=True):
            st.session_state["sample_choice"] = os.path.join(BASE_DIR, "sample_melanoma.jpg")
    with sample_col2:
        if st.button("🟢 Sample 2\n(Nevus)", use_container_width=True):
            st.session_state["sample_choice"] = os.path.join(BASE_DIR, "sample_nevus.jpg")
    with sample_col3:
        if st.button("🟡 Sample 3\n(Keratosis)", use_container_width=True):
            st.session_state["sample_choice"] = os.path.join(BASE_DIR, "sample_bkl.jpg")

# Determine which image source is active
active_image_bytes = None
image_source_label = ""

if uploaded_file is not None:
    active_image_bytes = uploaded_file.getvalue()
    image_source_label = f"Uploaded File ({uploaded_file.name})"
    # Clear sample selection if user uploaded a file
    if "sample_choice" in st.session_state:
        del st.session_state["sample_choice"]
elif "sample_choice" in st.session_state and os.path.exists(st.session_state["sample_choice"]):
    sample_path = st.session_state["sample_choice"]
    with open(sample_path, "rb") as f:
        active_image_bytes = f.read()
    image_source_label = f"Clinical Demo: {os.path.basename(sample_path)}"


# --------------------------------------------------------------------------------
# Core Pipeline Execution
# --------------------------------------------------------------------------------
if active_image_bytes is not None:
    bgr = inf.decode(active_image_bytes)
    
    if bgr is None:
        st.error("Failed to decode the image file. Please upload a standard JPG or PNG.")
        st.stop()
        
    orig_shape = (bgr.shape[1], bgr.shape[0])
    
    # Optional Scanning Animation / Simulation
    if enable_scan_sim and "scanned" not in st.session_state:
        scan_placeholder = st.empty()
        with scan_placeholder.container():
            st.markdown("""
            <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.5rem;">📡</div>
                    <div>
                        <div style="font-weight: 700; color: #38bdf8;">Dermoscopic Preprocessing & Optical Scanning Active...</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Filtering artifacts, segmenting lesion boundaries, and extracting dual-trunk feature tensors.</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            prog_bar = st.progress(0)
            stages = [
                "Acquiring optical array...",
                "Executing DullRazor morphological hair filter...",
                "Applying Telea inpainting artifact reconstruction...",
                "Balancing chromatic illumination with Shades-of-Gray...",
                "Segmenting lesion contour & computing ABCDE metrics...",
                "Running dual EfficientNetV2-M + B4 inference..."
            ]
            for idx, stage in enumerate(stages):
                prog_bar.progress(int((idx + 1) / len(stages) * 100), text=stage)
                time.sleep(0.08)
        scan_placeholder.empty()

    # 1. Execute full preprocessing stages
    pipeline_data = pt.run_pipeline_stages(bgr)
    
    # 2. Compute ABCDE features
    abcde = pt.compute_abcde_features(pipeline_data['final_tensor'], pipeline_data['segmentation'])
    
    # 3. Model Inference
    with st.spinner("Processing through dual-backbone neural network..."):
        res = inf.classify(
            model=model,
            rgb=pipeline_data['final_tensor'],
            meta=encoded_meta,
            routing=routing_mode
        )

    # ----------------------------------------------------------------------------
    # Hero Diagnosis Card
    # ----------------------------------------------------------------------------
    pred_label = res['label']
    info = pt.CLASS_INFO.get(pred_label, {})
    pred_type = info.get('type', 'Unknown')
    mal_prob = res.get('malignant_probability', 0.0)
    
    if pred_type == 'Malignant':
        card_class = "malignant"
        badge_type = "badge-red"
        status_text = "Malignant Neoplasm Detected"
    elif pred_type == 'Precancerous':
        card_class = "precancerous"
        badge_type = "badge-amber"
        status_text = "Dysplastic / Pre-Cancerous"
    else:
        card_class = "benign"
        badge_type = "badge-green"
        status_text = "Benign Cutaneous Lesion"

    st.markdown(f"""
    <div class="diagnosis-hero {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <span class="badge-pill {badge_type}">{status_text}</span>
                <span class="badge-pill badge-blue">Routing: {res['routing'].upper()}</span>
                <div class="diag-title" style="color: {info.get('color', '#38bdf8')};">
                    {info.get('name', pred_label)} ({pred_label})
                </div>
                <div style="color: #cbd5e1; font-size: 1rem; max-width: 800px; margin-top: 6px;">
                    {info.get('desc', '')}
                </div>
            </div>
            <div style="display: flex; gap: 14px; flex-wrap: wrap;">
                <div class="metric-box">
                    <div class="metric-value" style="color: {'#ef4444' if mal_prob > 0.4 else '#10b981'};">
                        {mal_prob:.1%}
                    </div>
                    <div class="metric-label">Malignancy Risk</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: #38bdf8;">
                        {res['probs'][pred_label]:.1%}
                    </div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: {abcde['badge_color']};">
                        {abcde['abc_badge']}
                    </div>
                    <div class="metric-label">ABCDE Score</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------
    # Interactive Tabs: Multi-Disciplinary Analysis
    # ----------------------------------------------------------------------------
    tab_scan, tab_abcde, tab_probs, tab_hierarchy, tab_guide, tab_report = st.tabs([
        "🔬 Preprocessing & Scanning Process",
        "📐 ABCDE Lesion Morphometrics",
        "📊 7-Class Probability Distribution",
        "🌳 Hierarchical Decision Tree",
        "📖 Dermatological Reference Atlas",
        "📄 Clinical Summary Report"
    ])

    # ----------------------------------------------------------------------------
    # TAB 1: Preprocessing & Scanning Process
    # ----------------------------------------------------------------------------
    with tab_scan:
        st.markdown("### 🔬 Multi-Stage Dermoscopic Preprocessing Pipeline")
        st.caption(
            "High-grade dermoscopy classification requires systematic removal of extraneous artifacts "
            "(hairs, air bubbles, skin oil glare) and chromatic standardization across different dermatoscopes."
        )
        
        # Pipeline Metrics
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Original Resolution", f"{orig_shape[0]} × {orig_shape[1]} px")
        pm2.metric("Hair Occlusion Density", f"{pipeline_data['hair_density']:.2f}%")
        pm3.metric("Standardized Resampling", f"{inf.CFG['img_size']} × {inf.CFG['img_size']} px")
        pm4.metric("Color Correction", f"Shades of Gray (p=6)")

        st.markdown("#### Intermediate Transformation Stages")
        st_c1, st_c2, st_c3, st_c4 = st.columns(4)
        
        with st_c1:
            st.markdown("**1. Raw Input**")
            st.image(pipeline_data['raw_rgb'], use_container_width=True, caption=f"Original ({orig_shape[0]}x{orig_shape[1]})")
            st.caption("Unmodified dermoscopy capture.")
            
        with st_c2:
            st.markdown("**2. Hair Artifact Mask**")
            st.image(pipeline_data['hair_mask'], use_container_width=True, caption=f"Blackhat Filter (thr={inf.CFG['hair_thr']})")
            st.caption("Morphological hair isolation.")

        with st_c3:
            st.markdown("**3. DullRazor Inpainting**")
            st.image(pipeline_data['inpainted_rgb'], use_container_width=True, caption="Telea 3-px Inpainting")
            st.caption("Occlusions removed cleanly.")

        with st_c4:
            st.markdown("**4. Color Normalization**")
            st.image(pipeline_data['final_tensor'], use_container_width=True, caption="Model Input (320x320 RGB)")
            st.caption("Chromatic illumination balanced.")

        st.markdown("---")
        st.markdown("#### 🎯 Before vs. After Preprocessing Comparison")
        comp1, comp2 = st.columns(2)
        with comp1:
            st.markdown("**Before: Raw Dermoscopic Capture**")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Contains hair occlusions, uneven illumination, and irregular aspect ratio.")
        with comp2:
            st.markdown("**After: Segmented & Chromatically Normalized Tensor**")
            st.image(pipeline_data['segmentation']['overlay_rgb'], use_container_width=True)
            st.caption("Lesion boundary delineated in blue, bounding box in gold, hairs in उन्हें removed.")

    # ----------------------------------------------------------------------------
    # TAB 2: Clinical ABCDE Lesion Morphometrics
    # ----------------------------------------------------------------------------
    with tab_abcde:
        st.markdown("### 📐 Objective Clinical ABCDE Morphometric Analysis")
        st.caption(
            "Computer vision algorithms extract quantitative shape, boundary, and pigment variation indicators "
            "mirroring the standard dermatological ABCDE rules for melanoma screening."
        )

        abc_c1, abc_c2 = st.columns([1.1, 0.9])
        
        with abc_c1:
            st.markdown("#### Visual Boundary Segmentation")
            st.image(pipeline_data['segmentation']['overlay_rgb'], use_container_width=True, caption="Cyan: Contour boundary | Orange: ROI Bounding Box")
            st.image(pipeline_data['segmentation']['mask'], use_container_width=True, caption="Binary Lesion Mask (Used for Morphometrics)")

        with abc_c2:
            st.markdown("#### Morphometric Parameters")
            
            st.markdown(f"""
            <div class="abcde-card">
                <div style="display: flex; justify-content: space-between;">
                    <strong>[A] Asymmetry Index</strong>
                    <span style="color: {'#ef4444' if abcde['asymmetry_score'] > 40 else '#10b981'}; font-weight: 700;">
                        {abcde['asymmetry_score']:.1f}% ({abcde['asymmetry_rating']})
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                    Measures symmetry across orthogonal horizontal and vertical axes. Melanomas typically display pronounced asymmetry.
                </div>
            </div>

            <div class="abcde-card">
                <div style="display: flex; justify-content: space-between;">
                    <strong>[B] Border Irregularity</strong>
                    <span style="color: {'#ef4444' if abcde['border_score'] > 35 else '#10b981'}; font-weight: 700;">
                        {abcde['border_score']:.1f}% ({abcde['border_rating']})
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                    Compactness ratio (P² / 4πA) = <strong>{abcde['compactness']:.2f}</strong>. Scalloped, notched, or ragged borders score higher.
                </div>
            </div>

            <div class="abcde-card">
                <div style="display: flex; justify-content: space-between;">
                    <strong>[C] Color Variegation</strong>
                    <span style="color: {'#ef4444' if abcde['color_score'] > 45 else '#10b981'}; font-weight: 700;">
                        {abcde['color_score']:.1f}% ({abcde['color_rating']})
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                    Dispersion across RGB channels = <strong>{abcde['color_std']:.1f}</strong>. Non-uniform pigmentation with multiple hues (brown, black, red, white) elevates risk.
                </div>
            </div>

            <div class="abcde-card">
                <div style="display: flex; justify-content: space-between;">
                    <strong>[D] Diameter & Coverage</strong>
                    <span style="color: #38bdf8; font-weight: 700;">
                        ~{abcde['est_diameter_mm']:.1f} mm ({abcde['coverage_pct']:.1f}% Area)
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                    Equivalent pixel diameter: {abcde['diameter_px']:.0f} px. Lesions > 6 mm historically warrant closer scrutiny under the ABCDE criteria.
                </div>
            </div>

            <div class="abcde-card" style="border-left-color: {abcde['badge_color']};">
                <div style="display: flex; justify-content: space-between;">
                    <strong>[E] Composite Morphologic Rating</strong>
                    <span style="color: {abcde['badge_color']}; font-weight: 700;">
                        {abcde['abc_level']} ({abcde['composite_abc_score']:.1f}/100)
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                    Integrated geometric score synthesized from Asymmetry, Border, and Color irregularity factors.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------
    # TAB 3: 7-Class Probability Distribution
    # ----------------------------------------------------------------------------
    with tab_probs:
        st.markdown("### 📊 Complete 7-Class Posterior Probability Distribution")
        st.caption(
            "Probabilities computed across all ISIC 2019 diagnostic classes using the selected routing mechanism."
        )

        prob_df = pd.DataFrame([
            {
                "Class Code": c,
                "Diagnostic Name": pt.CLASS_INFO[c]['name'],
                "Lesion Category": pt.CLASS_INFO[c]['type'],
                "Clinical Risk": pt.CLASS_INFO[c]['risk'],
                "Probability": prob,
                "Confidence (%)": f"{prob * 100:.2f}%"
            }
            for c, prob in sorted(res['probs'].items(), key=lambda x: x[1], reverse=True)
        ])

        col_chart, col_table = st.columns([1.2, 1.0])
        
        with col_chart:
            st.markdown("#### Posterior Probabilities")
            # Sort series for horizontal bar chart
            chart_series = pd.Series(
                {f"{pt.CLASS_INFO[c]['name']} ({c})": prob for c, prob in res['probs'].items()}
            ).sort_values(ascending=True)
            st.bar_chart(chart_series, color="#0ea5e9")

        with col_table:
            st.markdown("#### Detailed Probability Breakdown")
            for _, row in prob_df.iterrows():
                c_code = row["Class Code"]
                c_info = pt.CLASS_INFO[c_code]
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; margin-bottom: 6px; background: rgba(30, 41, 59, 0.45); border-radius: 8px; border-left: 3px solid {c_info['color']};">
                    <div>
                        <strong>{row['Diagnostic Name']}</strong> <span style="color: #94a3b8; font-size: 0.85rem;">({c_code})</span>
                        <div style="font-size: 0.78rem; color: #94a3b8;">{row['Lesion Category']} • {row['Clinical Risk']} Risk</div>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {'#38bdf8' if row['Probability'] > 0.2 else '#94a3b8'};">
                        {row['Confidence (%)']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ----------------------------------------------------------------------------
    # TAB 4: Hierarchical Decision Tree
    # ----------------------------------------------------------------------------
    with tab_hierarchy:
        st.markdown("### 🌳 Hierarchical Decision Network Traversal")
        st.caption(
            "Rather than treating all 7 classes uniformly, the model utilizes a clinically intuitive decision hierarchy: "
            "first separating Benign vs Malignant, then classifying specific histopathological subtypes."
        )

        nodes = res.get('nodes', {})
        
        # Display Decision Nodes as interactive visual cards
        hn1, hn2, hn3 = st.columns(3)
        
        with hn1:
            st.markdown("#### Node 1: Root Decision")
            root_dist = nodes.get('root', {})
            b_prob = root_dist.get('Benign', 0.0)
            m_prob = root_dist.get('Malignant', 0.0)
            
            st.markdown(f"""
            <div class="medical-card" style="border-top: 4px solid {'#ef4444' if m_prob > b_prob else '#10b981'};">
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">Root Node (Benign / Malignant)</div>
                <div style="margin-bottom: 6px;">
                    <span style="color: #10b981; font-weight: 600;">Benign:</span> {b_prob:.1%}
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="color: #ef4444; font-weight: 600;">Malignant:</span> {m_prob:.1%}
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8;">
                    <strong>Decision:</strong> {'🚨 Traverses to Malignant Subtree' if m_prob > b_prob else '✅ Traverses to Benign Subtree'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with hn2:
            st.markdown("#### Node 2: Malignant Subtree")
            mal_dist = nodes.get('mal', {})
            epi_dist = nodes.get('epi', {})
            
            st.markdown(f"""
            <div class="medical-card">
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">Malignant Sub-Branches</div>
                <div style="font-size: 0.88rem; margin-bottom: 4px;">
                    <strong>Node 'mal' (Melanoma vs Epidermal):</strong><br>
                    • MEL: {mal_dist.get('MEL', 0.0):.1%}<br>
                    • Epidermal: {mal_dist.get('Epidermal', 0.0):.1%}
                </div>
                <div style="font-size: 0.88rem; margin-top: 8px;">
                    <strong>Node 'epi' (BCC vs SCC):</strong><br>
                    • BCC: {epi_dist.get('BCC', 0.0):.1%}<br>
                    • SCC: {epi_dist.get('SCC', 0.0):.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with hn3:
            st.markdown("#### Node 3: Benign Subtree")
            ben_dist = nodes.get('ben', {})
            ker_dist = nodes.get('ker', {})
            
            st.markdown(f"""
            <div class="medical-card">
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">Benign Sub-Branches</div>
                <div style="font-size: 0.88rem; margin-bottom: 4px;">
                    <strong>Node 'ben' (DF vs NV vs Keratosis):</strong><br>
                    • DF: {ben_dist.get('DF', 0.0):.1%}<br>
                    • NV: {ben_dist.get('NV', 0.0):.1%}<br>
                    • Keratosis: {ben_dist.get('Keratosis', 0.0):.1%}
                </div>
                <div style="font-size: 0.88rem; margin-top: 8px;">
                    <strong>Node 'ker' (AK vs BKL):</strong><br>
                    • AK: {ker_dist.get('AK', 0.0):.1%}<br>
                    • BKL: {ker_dist.get('BKL', 0.0):.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Clinical Tree Traversal Path")
        st.code(f"""
[Input Dermoscopy: {orig_shape[0]}x{orig_shape[1]} px]
   │
   ▼
[Root Node: Malignant={mal_prob:.1%}, Benign={1 - mal_prob:.1%}]
   │
   ├── (Path Followed: {res['routing'].upper()})
   │
   └──▶ Final Label: {pred_label} ({info.get('name', '')}) [Confidence: {res['probs'][pred_label]:.1%}]
        """, language="text")

    # ----------------------------------------------------------------------------
    # TAB 5: Dermatological Reference Atlas
    # ----------------------------------------------------------------------------
    with tab_guide:
        st.markdown("### 📖 ISIC 2019 Dermatological Reference Atlas")
        st.caption(
            "Comprehensive clinical profiles, dermoscopic hallmarks, and management guidelines for all 7 target classes."
        )

        for code, c_data in pt.CLASS_INFO.items():
            with st.expander(f"{c_data['name']} ({code}) — [{c_data['type']} / {c_data['risk']} Risk]", expanded=(code == pred_label)):
                c_col1, c_col2 = st.columns([1.2, 1.0])
                with c_col1:
                    st.markdown(f"**Pathological Overview:**\n{c_data['desc']}")
                    st.markdown(f"**Key Dermoscopic Hallmarks:**\n`{c_data['hallmarks']}`")
                with c_col2:
                    st.markdown(f"**Recommended Clinical Action:**\n{c_data['action']}")
                    st.markdown(f"**Diagnostic Category:** `{c_data['type']}` | **Risk Tier:** `{c_data['risk']}`")

    # ----------------------------------------------------------------------------
    # TAB 6: Clinical Summary Report Generator
    # ----------------------------------------------------------------------------
    with tab_report:
        st.markdown("### 📄 Clinical Triage & Consultation Summary Report")
        st.caption("Standardized clinical report ready for medical records, peer review, or patient file documentation.")
        
        report_text = pt.generate_clinical_report(
            res=res,
            abcde=abcde,
            patient_meta=patient_meta,
            orig_shape=orig_shape
        )
        
        st.text_area("Generated Consultation Report", value=report_text, height=380)
        
        st.download_button(
            label="💾 Download Clinical Report (.txt)",
            data=report_text,
            file_name=f"dermoscan_report_{pred_label}_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True
        )

else:
    # ----------------------------------------------------------------------------
    # Empty State: Welcome Screen with Visual Guides
    # ----------------------------------------------------------------------------
    st.markdown("""
    <div class="medical-card" style="text-align: center; padding: 40px 20px;">
        <div style="font-size: 3.5rem; margin-bottom: 12px;">🔬</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
            Ready to Analyze Dermoscopic Lesion
        </div>
        <div style="color: #94a3b8; max-width: 600px; margin: 0 auto 24px auto; line-height: 1.6;">
            Upload a dermoscopic macro photograph using the file selector above, or click one of the 
            <strong>Clinical Demo Samples</strong> to test the real-time preprocessing scanning pipeline immediately.
        </div>
        <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
            <span class="badge-pill badge-blue">✓ DullRazor Hair Inpainting</span>
            <span class="badge-pill badge-green">✓ Shades-of-Gray Illumination</span>
            <span class="badge-pill badge-amber">✓ Automated ABCDE Morphometrics</span>
            <span class="badge-pill badge-blue">✓ Dual EfficientNet Fusion</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🌟 Key Diagnostic Capabilities:")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        st.markdown("""
        **1. Preprocessing Scanning Engine**
        - Automatic morphological hair artifact extraction
        - Telea 3-pixel border inpainting
        - Median noise filtering & Shades-of-Gray correction
        """)
        
    with f_col2:
        st.markdown("""
        **2. Clinical ABCDE Extraction**
        - Asymmetry index across orthogonal planes
        - Border compactness ($P^2 / 4\pi A$)
        - Color variegation and dispersion metrics
        """)

    with f_col3:
        st.markdown("""
        **3. Hierarchical Deep Learning**
        - EfficientNetV2-M + EfficientNet-B4 dual trunk
        - 5-node decision tree with clinical routing
        - Optional patient age, sex, and body site fusion
        """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem; padding: 12px;'>"
    "DermoScan AI Research System • ISIC 2019 Benchmark • Powered by TensorFlow & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
