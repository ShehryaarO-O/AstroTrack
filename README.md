# 🚀 AstroTrack

AstroTrack is a data-driven project focused on the orbital analysis of asteroids using real-world observational data. It integrates astrophysical modeling, statistical inference, and deep learning to impute missing values, engineer orbital features, detect anomalies, and predict object classes — all while respecting the scientific rigor of the domain.

---

## 📦 Dataset Overview

The dataset comprises observations of 4,500+ asteroids with 22 orbital features:

- 3 categorical and 19 numerical columns
- Key features include Mean Anomaly, Ascending Node Longitude, Perihelion Argument, Velocity, Distance, and Epoch Times
- Contains ~25–30% missing data

---

## 🔍 Exploratory Data Analysis

- Explored inter-feature correlations to guide imputation
- Analyzed missing value patterns based on temporal ordering
- Used domain insights to validate statistical relationships

---

## 🧮 Feature Imputation (Physics-Guided)

AstroTrack reconstructs missing values using orbital mechanics, not just statistics.

- **Time features**: Parsed Epoch timestamps, unified into a standard time column
- **Orbital geometry**: Recovered Mean Motion, Semi-Major Axis, Eccentricity, and Inclination using canonical orbital formulas
- **Velocity & distance**: Unified units across multiple representations
- **Temporal dynamics**: Imputed Mean Anomaly from Osculation and Perihelion time
- **Spatial parameters**: Regressed Perihelion Argument and Ascending Node Longitude
- **Orbital uncertainty**: Reconstructed using heliocentric distance, vis-viva equation, and perturbation estimates

📄 **Detailed equations, derivations, and physics-based imputations are available in the Report.pdf file.**

---

## 🧰 Feature Engineering Summary

Derived astrophysically meaningful features:

- Eccentricity, Inclination, Orbital Period, Heliocentric Distance
- Specific Angular Momentum, Orbital Energy, Escape Velocity
- True Anomaly, Synodic Period

---

## ⚖️ Data Normalization & Balancing

- Used `StandardScaler` and `RobustScaler` based on feature distributions
- Used weighted loss during training to address class imbalance

---

## 🧠 Model Architecture

Custom Deep Neural Network:

- Layers: 3 Dense (ReLU), Dropout, Output
- Initialization: He Normal
- Loss: Weighted Binary Cross-Entropy
- Tuning: Keras Tuner (dropout, neurons, learning rate)
- Validation: K-Fold Cross-Validation

**Performance:**
- F1 Score: 91.3%
- Accuracy: 84.3%
- Supported by AUC-ROC and confusion matrix

---

## 🧪 Anomaly Detection

### 1. Isolation Forest
- Contamination: 5%
- Detected: 227 anomalies

### 2. Custom Rule-Based
- Flagged objects with Tisserand’s T < 3 (potential comets)
- Flagged objects with uncertainty score ≥ 7
- Detected 140 anomalies
- Compared and analyzed overlap with Isolation Forest output

---

## 👨‍💻 Authors

Built by Shehryaar and Nayandeep — blending our love for astronomy, physics, and machine learning into a unified inference pipeline.
