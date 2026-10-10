# Hidden Markov Model for Crop Health State Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://sqlite.org/)
[![Model](https://img.shields.io/badge/Model-Gaussian_HMM-success.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

This standalone project detects **hidden crop-health states** (*Healthy*, *Stressed*, and *Disease-Suspected*) from weekly observation sequences of vegetation index (**NDVI**) and weather parameters (temperature, relative humidity, rainfall, and soil moisture) using a **Gaussian Hidden Markov Model (HMM)**.

---

## 🌟 Highlights & Key Features

- **Sequential Latent State Inference**: Unlike static point classifiers, HMM accounts for temporal state persistence and transition dynamics across the entire crop growing cycle.
- **Pure-NumPy/SciPy Architecture**: Hand-crafted, numerically stable implementation of:
  - **Viterbi Algorithm** (in log-space) for finding the optimal hidden state sequence.
  - **Forward-Backward Algorithm** for posterior marginal state probabilities $P(S_t = i \mid O_{1:T})$.
  - **Baum-Welch (EM)** for unsupervised parameter re-estimation.
- **Database & Benchmark Provenance**: Connected to local SQLite database (`crop_health.db`) and CSV records (`dataset.csv`, `kaggle_crop_health_dataset.csv`) sourced from online remote-sensing and Kaggle agro-climatic benchmarks.
- **Multi-Panel Visual Diagnostics**: Generates publication-ready diagnostic charts (`crop_health_hmm_analysis.png`) and ASCII progress bars in terminal.
- **Interactive Web Dashboard**: Client-side dashboard (`index.html`, `style.css`, `script.js`) for real-time scenario simulation and sequence editing.

---

## 🧠 Mathematical Formulation

An HMM is defined by the parameter tuple $\lambda = (A, B, \pi)$:

### 1. Hidden State Space
$$S = \{S_0: \text{Healthy}, \; S_1: \text{Stressed}, \; S_2: \text{Disease-Suspected}\}$$

### 2. Observation Vector
At each weekly timestep $t$, the observation vector $O_t \in \mathbb{R}^4$ consists of:
$$O_t = [\text{NDVI}_t, \; \text{Temperature}_t \; (^\circ\text{C}), \; \text{Relative Humidity}_t \; (\%), \; \text{Soil Moisture}_t \; (\%)]$$

### 3. Initial State Distribution ($\pi$)
$$\pi_i = P(S_1 = i), \quad \sum_{i=0}^2 \pi_i = 1$$
Prior beliefs reflect that crops overwhelmingly germinate in healthy status:
$$\pi \approx [0.70, \; 0.25, \; 0.05]$$

### 4. State Transition Matrix ($A$)
$$A_{ij} = P(S_{t+1} = j \mid S_t = i)$$

$$\begin{pmatrix}
P(H \to H) & P(H \to S) & P(H \to D) \\
P(S \to H) & P(S \to S) & P(S \to D) \\
P(D \to H) & P(D \to S) & P(D \to D)
\end{pmatrix} =
\begin{pmatrix}
0.908 & 0.064 & 0.028 \\
0.089 & 0.845 & 0.066 \\
0.176 & 0.000 & 0.824
\end{pmatrix}$$

### 5. Multivariate Gaussian Emission Distributions ($B$)
For state $k$, the emission density is modeled as a continuous Gaussian:
$$b_k(O_t) = \mathcal{N}(O_t \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{D/2}|\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(O_t - \mu_k)^T \Sigma_k^{-1} (O_t - \mu_k)\right)$$

| State | Mean NDVI ($\mu_1$) | Mean Temp ($\mu_2$) | Mean Humidity ($\mu_3$) | Mean Soil Moisture ($\mu_4$) | Agronomic Interpretation |
|---|---|---|---|---|---|
| **Healthy** | $0.59$ | $26.3^\circ\text{C}$ | $69.9\%$ | $48.9\%$ | High canopy vigor, balanced moisture |
| **Stressed** | $0.48$ | $28.4^\circ\text{C}$ | $44.4\%$ | $23.2\%$ | Moisture deficit, thermal leaf curling |
| **Disease-Suspected** | $0.45$ | $27.6^\circ\text{C}$ | $90.9\%$ | $62.2\%$ | Heavy humidity, canopy necrosis/lesions |

### 6. Decoding with the Viterbi Algorithm
To prevent numerical underflow over long sequences, dynamic programming operates in log-space:
$$\omega_1(i) = \log \pi_i + \log b_i(O_1)$$
$$\omega_t(j) = \max_{0 \le i \le 2} \left[ \omega_{t-1}(i) + \log A_{ij} \right] + \log b_j(O_t)$$
$$\psi_t(j) = \arg\max_{0 \le i \le 2} \left[ \omega_{t-1}(i) + \log A_{ij} \right]$$

Optimal path reconstruction:
$$q^*_T = \arg\max_i \omega_T(i), \quad q^*_t = \psi_{t+1}(q^*_{t+1})$$

---

## 🗄️ Database & Dataset Structure

The SQLite database (`crop_health.db`) stores observations in table `weekly_observations`:

```sql
CREATE TABLE weekly_observations (
    record_id TEXT PRIMARY KEY,
    field_id TEXT,
    week INTEGER,
    date TEXT,
    crop_type TEXT,
    ndvi REAL,
    temperature_c REAL,
    humidity_pct REAL,
    rainfall_mm REAL,
    soil_moisture_pct REAL,
    observed_symptom TEXT,
    ground_truth_state TEXT
);
```

- **Master CSV**: `dataset.csv`
- **Benchmark Source CSV**: `kaggle_crop_health_dataset.csv`

---

## 🚀 Setup & Execution

### 1. Requirements
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### 2. Run Database Connectivity & HMM Analysis
```bash
python3 analyze.py
```

This will:
1. Connect to `crop_health.db` and query weekly sequences.
2. Fit the Gaussian HMM and estimate parameters $\pi$, $A$, $\mu$, $\Sigma$.
3. Execute Viterbi decoding across all historical fields.
4. Print ASCII timelines in the console.
5. Generate diagnostic visualization plots saved to `crop_health_hmm_analysis.png`.

### 3. Open the Interactive Web Dashboard
Open `index.html` directly in any web browser:
```bash
open index.html
```

---

## 📁 Repository Structure

```
crop-health-hmm/
├── .git/                               # Independent Git repository
├── dataset.csv                         # Master CSV dataset of weekly crop observations
├── kaggle_crop_health_dataset.csv      # Online Kaggle-sourced dataset
├── crop_health.db                      # SQLite database populated from CSV
├── hmm_model.py                        # Standalone Gaussian HMM (Forward, Backward, Viterbi, EM)
├── analyze.py                          # Terminal analysis script & plot generator
├── crop_health_hmm_analysis.png        # Diagnostic multi-panel visualization plot
├── index.html                          # Interactive dashboard UI
├── style.css                           # Modern CSS design system
├── script.js                           # Client-side HMM inference engine
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation & mathematical guide
```

---

## 📊 Sample Output (FIELD-04: Tomato Blight Progression)

```
[ TIMELINE: FIELD-04 (Tomato) ]
----------------------------------------------------------------------
Week   | NDVI     | Decoded State          | True Benchmark  
----------------------------------------------------------------------
W1    | 0.40 ████████░░░░░░░░░░░░ | 🟢 Healthy            | 🟢 Healthy
W2    | 0.52 ██████████░░░░░░░░░░ | 🟢 Healthy            | 🟢 Healthy
W3    | 0.65 █████████████░░░░░░░ | 🟢 Healthy            | 🟢 Healthy
W4    | 0.72 ██████████████░░░░░░ | 🟢 Healthy            | 🟢 Healthy
W5    | 0.66 █████████████░░░░░░░ | 🔴 Disease-Suspected  | 🔴 Disease-Suspected
W6    | 0.55 ███████████░░░░░░░░░ | 🔴 Disease-Suspected  | 🔴 Disease-Suspected
W7    | 0.44 ████████░░░░░░░░░░░░ | 🔴 Disease-Suspected  | 🔴 Disease-Suspected
W8    | 0.38 ███████░░░░░░░░░░░░░ | 🔴 Disease-Suspected  | 🔴 Disease-Suspected
W9    | 0.42 ████████░░░░░░░░░░░░ | 🟢 Healthy            | 🔴 Disease-Suspected
W10   | 0.48 █████████░░░░░░░░░░░ | 🟢 Healthy            | 🟡 Stressed
W11   | 0.54 ██████████░░░░░░░░░░ | 🟢 Healthy            | 🟡 Stressed
W12   | 0.50 ██████████░░░░░░░░░░ | 🟢 Healthy            | 🟢 Healthy
----------------------------------------------------------------------
```
