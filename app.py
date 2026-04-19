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
# Updated Prediction Block for app.py
if uploaded_file:
    mat = sio.loadmat(uploaded_file, squeeze_me=True, struct_as_record=False)
    
    if 'DS' in mat:
        p_data = mat['DS']
        # Extract raw signal and metadata
        signal = p_data.rawData[0] if isinstance(p_data.rawData, np.ndarray) else p_data.rawData
        fs = p_data.samplingRate[0] if isinstance(p_data.samplingRate, np.ndarray) else p_data.samplingRate
        rpm = p_data.RPM[0] if isinstance(p_data.RPM, np.ndarray) else p_data.RPM
        
        # 1. Run the massive extraction function you just provided
        features_dict = extract_signal_features(signal, fs, rpm=rpm, placement_name='DS')
        
        # 2. Add the 'asset_encoded' column (Model expects it)
        features_dict['asset_encoded'] = 0 
        
        # 3. Create DataFrame and FORCE the column order
        features_df = pd.DataFrame([features_dict])
        
        # List of columns exactly as they appear in your Notebook's training set
        # This list must have all 58 features in the right order
        column_order = [
            'mean', 'std', 'rms', 'peak', 'peak_to_peak', 'crest_factor', 'kurtosis', 
            'skewness', 'dominant_freq', 'max_psd', 'mean_psd', 'top_freq_1', 'top_freq_2', 
            'top_freq_3', 'top_psd_1', 'top_psd_2', 'top_psd_3', 'spectral_entropy', 
            'low_band_energy', 'mid_band_energy', 'high_band_energy', 'harmonic_ratio_1', 
            'harmonic_ratio_2', 'envelope_peak_freq', 'envelope_peak_amp', 'bpfo_amp', 
            'bpfo_harmonic_2_amp', 'bpfo_harmonic_3_amp', 'bpfi_amp', 'bpfi_harmonic_2_amp', 
            'bpfi_harmonic_3_amp', 'env_harmonic_1', 'env_harmonic_2', 'env_harmonic_3', 
            'very_low_band_energy', 'low_mid_band_energy', 'mid_high_band_energy', 
            'very_high_band_energy', 'wavelet_energy_1', 'wavelet_energy_2', 
            'wavelet_energy_3', 'wavelet_energy_4', 'window_rms_std', 'window_rms_max', 
            'is_ds', 'is_fs', 'ds_weighted_rms', 'fs_weighted_rms', 'ds_weighted_kurtosis', 
            'fs_weighted_kurtosis', 'ds_weighted_peak', 'fs_weighted_peak', 
            'placement_interaction_rms', 'placement_interaction_kurtosis', 
            'sampling_rate', 'rpm', 'asset_encoded'
        ]
        
        # Filter and Reorder
        features_final = features_df[column_order]

        # 4. Now the scaler will work!
        feat_scaled = scaler.transform(features_final)

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
