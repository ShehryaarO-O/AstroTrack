# 🚀 AstroTrack

AstroTrack is a research-grade data science and astronomy project focused on the orbital analysis of asteroids using real-world observational data. It combines astrophysical modeling, statistical inference, and deep learning to impute missing values, engineer orbital features, detect anomalies, and predict object classes — all while respecting the scientific nature of the domain.

---

## 📦 Dataset Overview

The dataset comprises observations of 4,500+ asteroids with 22 core orbital features:

- 3 Categorical & 19 Numerical columns
- Key features include: Mean Anomaly, Ascending Node Longitude, Perihelion Argument, Velocity, Distance, and Epoch Times
- Approximately **25–30%** missing data

---

## 🔍 Exploratory Data Analysis

- Visualized inter-feature relationships to guide imputation logic
- Analyzed missing value patterns based on date ordering
- Used astronomical insights to construct hypotheses and confirm statistical relationships

---

## 🧮 Feature Imputation (Mathematically Driven)

AstroTrack goes beyond basic imputation by reconstructing missing data with physics-based equations:

### 1. ⏱️ Time Features
- Derived `year`, `month`, `day` from `Epoch Close Approach`
- Forward-filled missing date values using ordered observations
- Created a unified **Unix timestamp** column

### 2. 🌀 Orbital Geometry

Used orbital mechanics formulas to recover missing values:

- **Mean Motion (n)** and **Semi-Major Axis (a)**:

  \[
  n = \sqrt{\frac{GM}{a^3}} \quad \Rightarrow \quad a = \left(\frac{GM}{n^2}\right)^{1/3}
  \]

- **Eccentricity (e)**:

  \[
  e = \frac{r_a - a}{a}
  \]

- **Inclination (i)** from Tisserand's Parameter:

  \[
  T_j = \frac{a_J}{a} + 2\cos(i) \sqrt{\frac{a(1 - e^2)}{a_J}} \Rightarrow
  i = \cos^{-1}\left(\frac{T_j - \frac{a_J}{a}}{2\sqrt{\frac{a(1 - e^2)}{a_J}}}\right)
  \]

- **MICE Imputer** used after formulaic recovery to refine feature correlation estimates.

### 3. 🚀 Velocity & Distance Normalization

- Standardized all units of 3 velocity and 4 distance columns
- Used inter-feature mapping (e.g., continuous to categorical and vice versa) to infer missing entries
- Averaged cleaned columns into unified numerical `velocity` and `distance`

### 4. ⌛ Temporal Dynamics

Applied:

\[
M = n(t - \tau)
\]

Where:
- \( M \) = Mean Anomaly
- \( t \) = Epoch Osculation
- \( \tau \) = Perihelion Time

- Imputed via linear regression + interpolation
- Categorical Orbital Period was mapped to numerical form via theoretical values and violin plot validation

### 5. 🧭 Spatial Parameters

- `Perihelion Argument` & `Ascending Node Longitude` imputed via regression
- Evaluated via Recursive SHAP value analysis to determine their final model relevance

---

### 6. ❓ Orbital Uncertainty (Reconstructed Physically)

Original uncertainty data was sparse and unreliable. Reconstructed using:

#### a. Heliocentric Distance (\( r \)) via Cosine Law:

\[
r = \sqrt{r_1^2 + r_2^2 - 2r_1r_2\cos(\theta)}
\]

#### b. Velocity (\( v \)) via Vis-Viva Equation:

\[
v = \sqrt{GM\left(\frac{2}{r} - \frac{1}{a}\right)}
\]

#### c. Compute Perturbations:

\[
\Delta r, \Delta v \Rightarrow \Delta P, \Delta \tau
\]

#### d. Final MPC-Based Uncertainty Index:

\[
U = \log_{10} \left( \Delta P^2 + \Delta \tau^2 \right)
\]

Mapped to:
- 0–3 → Low
- 4–5 → Medium
- 6+  → High

---

## 🧰 Feature Engineering Summary

Engineered physically significant features:

- Eccentricity, Inclination Angle
- Orbital Period (Numerical), Heliocentric Distance
- Specific Angular Momentum & Energy
- Escape Velocity, Velocity at Perihelion/Aphelion
- Synodic Period, True Anomaly

---

## ⚖️ Data Normalization & Balancing

- Used **StandardScaler** and **RobustScaler** depending on distribution
- Avoided SMOTE due to model choice (DNN) and literature-based evidence
- Balanced via **weighted loss functions** in training instead

---

## 🧠 Model Architecture

Custom Deep Neural Network (DNN):

- **Layers:** 3 Dense (ReLU) + Dropout + Output
- **Weight Initialization:** He Normal
- **Loss Function:** `tf.nn.weighted_cross_entropy_with_logits`
- **Tuning:** Keras-Tuner (neurons, dropout, learning rate, class weights)
- **Validation:** K-Fold Cross-Validation

### ✅ Performance

- **F1 Score:** 91.3%
- **Accuracy:** 84.3%
- Supported by confusion matrix & AUC-ROC curves

---

## 🧪 Anomaly Detection

### 1. 📦 Isolation Forest (Library-Based)

- Contamination Rate: 0.05  
- Detected: 227 anomalies

### 2. 🧠 Custom Rule-Based

- Flag 1: Tisserand’s \( T_j < 3 \) (likely cometary bodies)
- Flag 2: Uncertainty score ≥ 7 (high unpredictability)

- Detected: 140 anomalies  
- Compared overlap with Isolation Forest results

