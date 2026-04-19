import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import scipy.io as sio
from scipy.stats import kurtosis, skew
from scipy.signal import welch, hilbert
from scipy.fft import rfft, rfftfreq
import pywt
from sklearn.metrics import classification_report, confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="SCA Bearing Diagnosis Pro", layout="wide")
st.title("🛠️ Hybrid Transformer-XGBoost Diagnosis System")
st.markdown("---")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Ensure these names match your saved files exactly
    ae = tf.keras.models.load_model('models/transformer_ae.keras', compile=False)
    anomaly_clf = joblib.load('models/anomaly_clf.pkl')
    fault_clf = joblib.load('models/fault_clf.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return ae, anomaly_clf, fault_clf, scaler

ae, anomaly_clf, fault_clf, scaler = load_assets()

# --- NOTEBOOK FEATURE EXTRACTION (CELL 587) ---
def extract_signal_features(signal, fs, rpm=None, placement_name='DS'):
    signal = np.array(signal).flatten()
    if len(signal) == 0: return None
    
    # Time Domain
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    kurt = kurtosis(signal)
    
    # Frequency Domain
    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    
    # Envelope & FFT (Cell 587)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    env_freqs, env_psd = welch(envelope, fs=fs, nperseg=min(1024, len(envelope)))
    
    # Wavelet (Cell 587)
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    
    # Metadata for BPFI/BPFO
    shaft_freq = rpm / 60 if rpm is not None else 0
    bpfo_freq, bpfi_freq = 4.8 * shaft_freq, 5.2 * shaft_freq

    def get_band_amp(f_arr, a_arr, target):
        indices = np.where((f_arr >= target - 5) & (f_arr <= target + 5))[0]
        return np.max(a_arr[indices]) if len(indices) > 0 else 0

    # Build exact dictionary from Notebook Cell 587
    # Note: Use DS for local UI testing by default
    is_ds = 1 if placement_name == 'DS' else 0
    is_fs = 1 if placement_name == 'FS' else 0

    return {
        'mean': np.mean(signal), 'std': np.std(signal), 'rms': rms, 'peak': peak,
        'peak_to_peak': np.ptp(signal), 'crest_factor': peak/(rms+1e-8),
        'kurtosis': kurt, 'skewness': skew(signal),
        'dominant_freq': freqs[np.argmax(psd)], 'max_psd': np.max(psd), 'mean_psd': np.mean(psd),
        'top_freq_1': freqs[np.argsort(psd)[-1]], 'top_freq_2': freqs[np.argsort(psd)[-2]],
        'top_freq_3': freqs[np.argsort(psd)[-3]], 'top_psd_1': np.sort(psd)[-1],
        'top_psd_2': np.sort(psd)[-2], 'top_psd_3': np.sort(psd)[-3],
        'spectral_entropy': -np.sum((psd/np.sum(psd))*np.log2(psd/np.sum(psd)+1e-8)),
        'low_band_energy': np.sum(psd[(freqs >= 0) & (freqs < 100)]),
        'mid_band_energy': np.sum(psd[(freqs >= 100) & (freqs < 500)]),
        'high_band_energy': np.sum(psd[(freqs >= 500) & (freqs < 2000)]),
        'harmonic_ratio_1': np.sort(psd)[-2]/(np.sort(psd)[-1]+1e-8),
        'harmonic_ratio_2': np.sort(psd)[-3]/(np.sort(psd)[-1]+1e-8),
        'envelope_peak_freq': env_freqs[np.argmax(env_psd)], 'envelope_peak_amp': np.max(env_psd),
        'bpfo_amp': get_band_amp(rfftfreq(len(signal), 1/fs), np.abs(rfft(signal)), bpfo_freq),
        'bpfo_harmonic_2_amp': get_band_amp(rfftfreq(len(signal), 1/fs), np.abs(rfft(signal)), 2*bpfo_freq),
        'bpfo_harmonic_3_amp': get_band_amp(rfftfreq(len(signal), 1/fs), np.abs(rfft(signal)), 3*bpfo_freq),
        'bpfi_amp': get_band_amp(rfftfreq(len(signal), 1/fs), np.abs(rfft(signal)), bpfi_freq),
        'bpfi_harmonic_2_amp': get_band_amp(rfftfreq(len(signal), 1/fs), np.abs(rfft(signal)), 2*bpfi_freq),
        'bpfi_harmonic_3_amp': get_band_amp(rfftfreq(len(signal), 1/fs), np.abs(rfft(signal)), 3*bpfi_freq),
        'env_harmonic_1': get_band_amp(env_freqs, env_psd, env_freqs[np.argmax(env_psd)]),
        'env_harmonic_2': get_band_amp(env_freqs, env_psd, 2*env_freqs[np.argmax(env_psd)]),
        'env_harmonic_3': get_band_amp(env_freqs, env_psd, 3*env_freqs[np.argmax(env_psd)]),
        'very_low_band_energy': np.sum(psd[(freqs >= 0) & (freqs < 50)]),
        'low_mid_band_energy': np.sum(psd[(freqs >= 50) & (freqs < 150)]),
        'mid_high_band_energy': np.sum(psd[(freqs >= 150) & (freqs < 500)]),
        'very_high_band_energy': np.sum(psd[(freqs >= 500) & (freqs < 3000)]),
        'wavelet_energy_1': np.sum(np.square(coeffs[0])), 'wavelet_energy_2': np.sum(np.square(coeffs[1])),
        'wavelet_energy_3': np.sum(np.square(coeffs[2])), 'wavelet_energy_4': np.sum(np.square(coeffs[3])),
        'window_rms_std': np.std(signal), 'window_rms_max': rms,
        'is_ds': is_ds, 'is_fs': is_fs, 'ds_weighted_rms': rms*is_ds, 'fs_weighted_rms': rms*is_fs,
        'ds_weighted_kurtosis': kurt*is_ds, 'fs_weighted_kurtosis': kurt*is_fs,
        'ds_weighted_peak': peak*is_ds, 'fs_weighted_peak': peak*is_fs,
        'placement_interaction_rms': (rms*is_ds)-(rms*is_fs),
        'placement_interaction_kurtosis': (kurt*is_ds)-(kurt*is_fs),
        'sampling_rate': fs, 'rpm': rpm if rpm is not None else 0, 'asset_encoded': 0, 'placement_encoded': 0
    }

# --- HYBRID PIPELINE LOGIC (CELL 619) ---
def run_hybrid_inference(features_dict, threshold=0.1652):
    # 1. Prepare Data
    df = pd.DataFrame([features_dict])
    # IMPORTANT: Ensure columns are in exact order used in Notebook Cell 592
    scaled = scaler.transform(df)
    
    # 2. Stage 1: Autoencoder Reconstruction Error
    recon = ae.predict(scaled, verbose=0)
    recon_error = np.mean(np.square(scaled - recon), axis=1)[0]
    
    # 3. Stage 2: XGBoost Anomaly Score
    anomaly_probs = anomaly_clf.predict_proba(df)[0]
    xgb_anomaly_score = anomaly_probs[1]
    
    # 4. HYBRID GATE (Cell 619 Logic)
    # Thresholds: recon_error > 0.165 or xgb > 0.85
    if (recon_error > threshold) or (xgb_anomaly_score > 0.85):
        fault_probs = fault_clf.predict_proba(df)[0]
        confidence = np.max(fault_probs)
        
        # Stricter confidence filter (0.80) from Cell 619
        if confidence < 0.80:
            return 0, recon_error, xgb_anomaly_score
        
        # Map 0,1,2 -> 1,2,3 (Inner, Ball, Outer)
        return int({0:1, 1:2, 2:3}[np.argmax(fault_probs)]), recon_error, xgb_anomaly_score
    
    return 0, recon_error, xgb_anomaly_score

# --- UI LOGIC ---
st.sidebar.header("Batch Diagnostics")
uploaded_files = st.file_uploader("Upload .mat files for Evaluation", type="mat", accept_multiple_files=True)

if uploaded_files:
    all_results = []
    with st.spinner("Analyzing signals using Transformer-XGBoost Hybrid Pipeline..."):
        for uploaded_file in uploaded_files:
            mat = sio.loadmat(uploaded_file, squeeze_me=True, struct_as_record=False)
            sensor = 'DS' if 'DS' in mat else 'FS'
            p_data = mat[sensor]
            
            # Ground Truth (For Metrics)
            true_label = int(p_data.label) if not isinstance(p_data.label, np.ndarray) else int(p_data.label[0])
            
            # Process
            feats = extract_signal_features(p_data.rawData, p_data.samplingRate, rpm=p_data.RPM, placement_name=sensor)
            pred, err, score = run_hybrid_inference(feats)
            
            all_results.append({'File': uploaded_file.name, 'True': true_label, 'Pred': pred, 'Error': round(err, 4)})

    res_df = pd.DataFrame(all_results)
    
    # --- PERFORMANCE DELIVERABLES (AS REQUESTED) ---
    st.subheader("📊 System Performance Report")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Classification Report**")
        report = classification_report(res_df['True'], res_df['Pred'], labels=[0,1,2,3], output_dict=True, zero_division=0)
        st.table(pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score', 'support']])
        
    with c2:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(res_df['True'], res_df['Pred'], labels=[0,1,2,3])
        st.write(cm)
        st.caption("Rows: Ground Truth | Columns: Predictions (0=Normal, 1=Inner, 2=Ball, 3=Outer)")

    st.dataframe(res_df)
