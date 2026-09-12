import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Cancer Prediction System",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for vibrant, colorful, and high-impact visual aesthetics
st.markdown("""
    <style>
    /* Gradient Banner Header */
    .header-card {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(135deg, #4f46e5, #7c3aed, #db2777);
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.3);
        text-align: center;
        margin-bottom: 25px;
    }
    .header-title { font-size: 2.4rem; font-weight: 900; color: #ffffff; text-shadow: 0 2px 10px rgba(0,0,0,0.3); margin: 0; }
    .header-sub { font-size: 1.1rem; color: #f1f5f9; font-weight: 500; margin-top: 8px; }
    
    /* Vibrant Input Card Containers */
    .section-tag {
        display: inline-block;
        padding: 6px 14px;
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        color: white; font-weight: 700; border-radius: 20px; font-size: 0.9rem; margin-bottom: 15px;
    }

    /* Result Cards */
    .result-safe {
        padding: 22px; border-radius: 14px;
        background: linear-gradient(135deg, #059669, #10b981);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.35);
        color: #ffffff; font-size: 1.4rem; font-weight: 800; text-align: center;
    }
    .result-danger {
        padding: 22px; border-radius: 14px;
        background: linear-gradient(135deg, #dc2626, #f43f5e);
        box-shadow: 0 8px 20px rgba(244, 63, 94, 0.35);
        color: #ffffff; font-size: 1.4rem; font-weight: 800; text-align: center;
    }

    /* Vibrant Gradient Submit Button */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899) !important;
        color: white !important; font-size: 1.2rem !important; font-weight: 800 !important;
        border: none !important; border-radius: 12px !important; padding: 14px !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.3s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 10px 25px rgba(236, 72, 153, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Vibrant Banner
st.markdown("""
    <div class="header-card">
        <div class="header-title">🔬 Tumor Cell Health Checker</div>
        <div class="header-sub">AI Machine Learning Diagnostic Pipeline | FastAPI & Streamlit</div>
    </div>
""", unsafe_allow_html=True)

# Feature Configuration (Label, Min, Max, Default, Step)
FEATURE_CONFIG = {
    "mean_radius": ("Cell Size (Mean Radius)", 6.0, 30.0, 14.12, 0.1),
    "mean_texture": ("Cell Texture (Mean Texture)", 9.0, 40.0, 19.28, 0.1),
    "mean_perimeter": ("Cell Border Length (Mean Perimeter)", 40.0, 190.0, 91.96, 0.1),
    "mean_area": ("Cell Total Area (Mean Area)", 140.0, 2500.0, 654.88, 1.0),
    "mean_smoothness": ("Cell Smoothness (Mean Smoothness)", 0.05, 0.25, 0.096, 0.001),
    "mean_compactness": ("Cell Density (Mean Compactness)", 0.02, 0.35, 0.104, 0.001),
    "mean_concavity": ("Cell Shape Depth (Mean Concavity)", 0.0, 0.45, 0.088, 0.001),
    "mean_concave_points": ("Cell Edge Points (Mean Concave Points)", 0.0, 0.25, 0.048, 0.001),
    "mean_symmetry": ("Cell Shape Symmetry (Mean Symmetry)", 0.1, 0.35, 0.181, 0.001),
    "mean_fractal_dimension": ("Cell Complexity (Fractal Dim)", 0.04, 0.10, 0.062, 0.001)
}

st.markdown('<div class="section-tag">⚙️ Enter Tumor Measurements</div>', unsafe_allow_html=True)
cols = st.columns(2)
inputs = {}

# Simple loop for sliders in 2 clean columns
for i, (key, (label, min_v, max_v, default_v, step_v)) in enumerate(FEATURE_CONFIG.items()):
    with cols[i % 2]:
        inputs[key] = st.slider(label, min_v, max_v, default_v, step_v)

st.write("")
if st.button("🚀 PREDICT CANCER DIAGNOSIS", use_container_width=True):
    result = None

    # Try FastAPI Backend Call
    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=inputs, timeout=3)
        if res.status_code == 200:
            result = res.json()
            st.caption("⚡ Live prediction served via FastAPI Backend")
    except Exception:
        # Local Model Fallback
        if os.path.exists("model.pkl"):
            data = joblib.load("model.pkl")
            full_feat = {c: inputs[c.replace(" ", "_")] if c.replace(" ", "_") in inputs else 0.0 for c in data["feature_names"]}
            scaled = data["scaler"].transform(pd.DataFrame([full_feat])[data["feature_names"]])
            pred = data["model"].predict(scaled)[0]
            probs = data["model"].predict_proba(scaled)[0]
            result = {
                "prediction": "Safe (No Cancer / Benign)" if pred == 1 else "Cancerous (Harmful / Malignant)",
                "confidence_score": round(float(np.max(probs)) * 100, 2),
                "probabilities": {"Benign": round(float(probs[1]) * 100, 2), "Malignant": round(float(probs[0]) * 100, 2)}
            }
            st.caption("ℹ️ Model Fallback Prediction")

    if result:
        st.divider()
        is_safe = "Safe" in result["prediction"] or "Benign" in result["prediction"]
        box_class = "result-safe" if is_safe else "result-danger"
        icon = "✅" if is_safe else "⚠️"
        
        st.markdown(f'<div class="{box_class}">{icon} DIAGNOSIS: {result["prediction"]} ({result["confidence_score"]}% Confidence)</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-tag">📊 Probability Breakdown</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Safe / No Cancer Chance (Benign)", f"{result['probabilities']['Benign']}%")
            st.progress(result['probabilities']['Benign'] / 100.0)
        with c2:
            st.metric("Cancerous Chance (Malignant)", f"{result['probabilities']['Malignant']}%")
            st.progress(result['probabilities']['Malignant'] / 100.0)
