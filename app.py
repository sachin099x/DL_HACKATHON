import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import scipy.io as sio
from scipy.stats import kurtosis, skew
from scipy.signal import welch

# --- Page Config ---
st.set_page_config(page_title="Bearing Fault Diagnosis", layout="wide")
st.title("🛠️ SCA Bearing Fault Diagnosis System")
st.markdown("Upload a `.mat` vibration signal file to analyze bearing health using the **Hybrid Transformer-XGBoost Pipeline**.")

# --- Load Models & Scaler ---
@st.cache_resource
def load_assets():
    # Ensure these files are in your /models folder on GitHub
    ae = tf.keras.models.load_model('models/transformer_ae.keras')
    anomaly_clf = joblib.load('models/anomaly_clf.pkl')
    fault_clf = joblib.load('models/fault_clf.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return ae, anomaly_clf, fault_clf, scaler

ae, anomaly_clf, fault_clf, scaler = load_assets()

# --- Feature Extraction Function (Matches your Notebook) ---
def extract_features(signal, fs):
    signal = np.array(signal).flatten()
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    kurt = kurtosis(signal)
    crest = peak / (rms + 1e-8)
    
    # Frequency domain
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    dom_freq = freqs[np.argmax(psd)]
    
    # Return features in the exact order expected by your model
    return pd.DataFrame([{
        'rms': rms, 'kurtosis': kurt, 'crest_factor': crest, 
        'dom_freq': dom_freq, 'peak': peak, 'mean': np.mean(signal)
        # ... Add all 58 features used in feature_columns from your notebook
    }])

# --- UI Sidebar & Upload ---
st.sidebar.header("Instructions")
st.sidebar.info("1. Upload a .mat file.\n2. The system extracts 58 features.\n3. Hybrid logic determines if it's a fault.")

uploaded_file = st.file_uploader("Choose a .mat file", type="mat")

if uploaded_file:
    # Load .mat file
    mat = sio.loadmat(uploaded_file, squeeze_me=True, struct_as_record=False)
    
    # Assuming standard DS placement for demo
    if 'DS' in mat:
        p_data = mat['DS']
        signal = p_data.rawData[0] if isinstance(p_data.rawData, np.ndarray) else p_data.rawData
        fs = p_data.samplingRate[0] if isinstance(p_data.samplingRate, np.ndarray) else p_data.samplingRate
        
        st.subheader("Signal Analysis")
        features = extract_features(signal, fs)
        st.write("Extracted Features (Preview):", features.head())
        
        # --- Prediction Logic (Strategy 2: Hybrid) ---
        # 1. Scaling
        feat_scaled = scaler.transform(features)
        
        # 2. Autoencoder Check
        recon = ae.predict(feat_scaled)
        recon_error = np.mean(np.square(feat_scaled - recon))
        
        # 3. XGBoost Anomaly Check
        is_anomaly_prob = anomaly_clf.predict_proba(features)[0][1]
        
        # Hybrid Gate (Based on your Strategy 2)
        if (recon_error > 0.32) or (is_anomaly_prob > 0.85):
            fault_probs = fault_clf.predict_proba(features)[0]
            fault_class = np.argmax(fault_probs)
            
            label_map = {0: "Inner Ring Fault", 1: "Ball Fault", 2: "Outer Ring Fault"}
            st.error(f"⚠️ ANOMALY DETECTED: {label_map[fault_class]}")
            st.metric("Classifier Confidence", f"{np.max(fault_probs)*100:.2f}%")
        else:
            st.success("✅ BEARING STATUS: NORMAL")
    else:
        st.warning("Could not find 'DS' sensor data in the uploaded file.")
