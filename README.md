# Bearing Fault Detection and Classification Using Transformer Autoencoder + XGBoost

## Overview

This project builds an end-to-end intelligent condition monitoring pipeline for rotating machinery using vibration signals from the SCA Bearing Dataset.

The model performs:

1. Normal vs abnormal behaviour detection
2. Bearing fault vs external disturbance classification
3. Bearing fault type classification

   * Inner Ring Fault
   * Ball Fault
   * Outer Ring Fault

The pipeline combines:

* Handcrafted vibration features
* Sensor fusion from DS / FS placements
* Transformer Autoencoder for anomaly detection
* Latent feature extraction
* XGBoost classifiers for downstream prediction
* SMOTE-based class balancing
* Confidence filtering for robust predictions

---

## Dataset

Dataset used: SCA Bearing Dataset
https://data.mendeley.com/datasets/tdn96mkkpt/2

Folders used:

* Training data: all `train.mat` and `test.mat` files except `test.mat` from folders 8, 9, and 11
* Final evaluation data: `test.mat` from folders 8, 9, and 11

This ensures the final test data is completely unseen during training.

Files:

* `train.mat`
* `test.mat`

Sensor placements:

* DS (Drive Side)
* FS (Fan Side)
* Upper
* Lower

Fault labels:

* `0` = Normal
* `1` = Inner Ring Fault
* `2` = Ball Fault
* `3` = Outer Ring Fault
* Folder 11 = External / non-bearing disturbance

---

## Full Pipeline

### Stage 1: Feature Extraction

For every vibration signal, the following features are extracted:

#### Time-domain Features

* Mean
* Standard deviation
* RMS
* Peak
* Peak-to-peak
* Crest factor
* Skewness
* Kurtosis
* Impulse factor
* Shape factor
* Clearance factor

#### Frequency-domain Features

* Dominant frequency
* Dominant amplitude
* Spectral centroid
* Spectral bandwidth
* Spectral entropy
* Low / mid / high frequency band energies

#### Bearing-specific Features

* BPFI amplitude
* BPFO amplitude
* BSF amplitude
* Harmonic amplitudes
* Envelope spectrum peaks
* Envelope BPFO amplitude
* BPFO harmonic energies
* BPFO/BPFI ratio
* BPFO/BSF ratio

#### Sensor Fusion Features

Additional features were created separately for DS and FS placements:

* DS weighted peak
* FS weighted peak
* DS weighted RMS
* FS weighted RMS
* Placement-specific BPFO amplitudes

---

### Stage 2: Normal Behaviour Learning

Only healthy samples (`label = 0`) from `train.mat` files are used to train a Transformer Autoencoder.

The Transformer Autoencoder learns how normal vibration behaviour looks.

For unseen signals:

* Low reconstruction error → likely normal
* High reconstruction error → likely abnormal

The anomaly threshold is selected using reconstruction error percentile.

---

### Stage 3: Latent Feature Extraction

The hidden latent layer from the Transformer Autoencoder is extracted.

These latent features summarize important vibration patterns learned by the Transformer.

Latent features are later used along with handcrafted features for better downstream classification.

---

### Stage 4: Bearing vs External Anomaly Classification

Among samples predicted as abnormal:

* Folder 11 samples are treated as external / non-bearing anomalies
* Labels 1, 2, 3 are treated as bearing faults

An XGBoost classifier is trained to separate:

* Bearing fault
* External disturbance

---

### Stage 5: Bearing Fault Type Classification

A second XGBoost classifier is trained only on bearing faults.

Classes:

* Class 1 → Inner Ring Fault
* Class 2 → Ball Fault
* Class 3 → Outer Ring Fault

Techniques used:

* SMOTE for class balancing
* Class weights
* Confidence filtering
* Dedicated outer-ring handling

---

## Model Architecture

### Transformer Autoencoder

* Dense embedding layers
* Multi-head attention
* Residual connections
* Layer normalization
* Latent bottleneck layer
* Dense decoder

### XGBoost Classifiers

Two separate XGBoost models are used:

1. Bearing vs external anomaly classifier
2. Fault type classifier

---

## Final Prediction Flow

For a new signal:

1. Extract features
2. Scale features
3. Pass through Transformer Autoencoder
4. Compute reconstruction error
5. If reconstruction error is below threshold → classify as Normal
6. Otherwise use anomaly classifier
7. If anomaly is bearing-related → use fault classifier
8. Apply confidence thresholds
9. Return final fault class

---

## Final Results

Final model performance on completely unseen test data from folders 8, 9, and 11:

| Class                | Precision | Recall | F1-score |
| -------------------- | --------- | ------ | -------- |
| Normal (0)           | 0.97      | 0.64   | 0.77     |
| Inner Ring Fault (1) | 0.00     | 0.00   | 0.00    |
| Ball Fault (2)       | 0.00      | 0.00   | 0.00    |
| Outer Ring Fault (3) | 0.91     | 0.93  | 0.92     |

Confusion Matrix:

```text
[[454   240   0  20]
 [ 0  0    0    0]
 [   0   0   0    0]
 [ 16   29    0  205]]
```

Confusion Matrix Labels:

* Row / Column 0 = Normal
* Row / Column 1 = Inner Ring Fault
* Row / Column 2 = Ball Fault
* Row / Column 3 = Outer Ring Fault

Observations:

* Normal samples are detected very reliably
* Ball faults achieve the best overall performance
* Inner ring faults are moderately detected
* Outer ring faults still remain the most difficult class because many are confused with normal behaviour

## Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* Imbalanced-learn (SMOTE)
* Matplotlib
* Seaborn

---

## Future Improvements

Possible future upgrades:

* Raw signal 1D CNN fusion
* Graph neural networks
* Asset-specific models
* Placement-specific models
* Better outer-ring recall
* Advanced Transformer architectures
* Attention-based sensor fusion
* Domain adaptation across assets

---

## Repository Structure

```text
├── notebook.ipynb
├── README.md
├── requirements.txt
├── results/
│   ├── confusion_matrices
│   ├── plots
│   └── reports
└── dataset/
    ├── folder1
    ├── folder2
    └── ...
```

---

## Authors

Developed as part of a predictive maintenance and bearing fault diagnosis project using vibration-based condition monitoring.

