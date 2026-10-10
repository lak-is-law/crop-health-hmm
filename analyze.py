import os
import sys
import sqlite3
import pandas as pd
import numpy as np

# Ensure matplotlib uses writable directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hmm_model import GaussianHMM

DB_NAME = os.path.join(SCRIPT_DIR, "crop_health.db")
CSV_NAME = os.path.join(SCRIPT_DIR, "dataset.csv")

def connect_db_and_load_data():
    """Connects to SQLite database and fetches weekly observations."""
    if os.path.exists(DB_NAME):
        print(f"[*] Connecting to SQLite database: '{DB_NAME}'...")
        conn = sqlite3.connect(DB_NAME)
        query = "SELECT * FROM weekly_observations ORDER BY field_id, week"
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"[*] Query executed successfully. Loaded {len(df)} records from 'weekly_observations'.\n")
        return df
    elif os.path.exists(CSV_NAME):
        print(f"[!] '{DB_NAME}' not found. Loading fallback from '{CSV_NAME}'...")
        df = pd.read_csv(CSV_NAME)
        return df
    else:
        print(f"[Error] Neither {DB_NAME} nor {CSV_NAME} found.")
        sys.exit(1)

def print_header(title):
    print("=" * 70)
    print(f"   {title}")
    print("=" * 70)

def print_ascii_timeline(field_id, crop_type, weeks, ndvis, decoded_states, true_states=None):
    """Renders a visual ASCII progress bar and state sequence timeline."""
    symbol_map = {"Healthy": "🟢 H", "Stressed": "🟡 S", "Disease-Suspected": "🔴 D"}
    print(f"\n[ TIMELINE: {field_id} ({crop_type}) ]")
    print("-" * 70)
    print(f"{'Week':<6} | {'NDVI':<8} | {'Decoded State':<22} | {'True Benchmark':<16}")
    print("-" * 70)
    for w, n, d, t in zip(weeks, ndvis, decoded_states, true_states or decoded_states):
        sym_d = symbol_map.get(d, d)
        sym_t = symbol_map.get(t, t)
        bar_len = int(n * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"W{w:<4} | {n:.2f} {bar} | {sym_d:<20} | {sym_t}")
    print("-" * 70)

def generate_visualizations(df, hmm, sample_field="FIELD-04", output_file=None):
    """Generates multi-panel diagnostic plots for HMM analysis."""
    if output_file is None:
        output_file = os.path.join(SCRIPT_DIR, "crop_health_hmm_analysis.png")
        
    print(f"\n[*] Generating HMM diagnostic visualizations for field {sample_field}...")
    field_df = df[df["field_id"] == sample_field].sort_values("week")
    features = ["ndvi", "temperature_c", "humidity_pct", "soil_moisture_pct"]
    X = field_df[features].values
    
    pred_res = hmm.predict_sequence(X)
    decoded_states = pred_res["decoded_states"]
    posteriors = pred_res["posteriors"]
    weeks = field_df["week"].values
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Hidden Markov Model: Crop Health Analysis ({sample_field} - {field_df['crop_type'].iloc[0]})",
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Observation Trajectories with Decoded State Background
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    p1, = ax1.plot(weeks, field_df["ndvi"], color="#16a34a", marker="o", linewidth=2.5, label="NDVI (Vegetation Index)")
    p2, = ax1_twin.plot(weeks, field_df["humidity_pct"], color="#0284c7", marker="s", linestyle="--", label="Humidity (%)")
    p3, = ax1_twin.plot(weeks, field_df["temperature_c"], color="#dc2626", marker="^", linestyle=":", label="Temperature (°C)")
    
    state_color_map = {"Healthy": "#bbf7d0", "Stressed": "#fef08a", "Disease-Suspected": "#fecaca"}
    for w, state in zip(weeks, decoded_states):
        ax1.axvspan(w - 0.5, w + 0.5, color=state_color_map.get(state, "#f3f4f6"), alpha=0.35, lw=0)
        
    ax1.set_xlabel("Growth Cycle (Week)", fontweight='bold')
    ax1.set_ylabel("NDVI", color="#16a34a", fontweight='bold')
    ax1_twin.set_ylabel("Weather Parameters (°C, %)", color="#0284c7", fontweight='bold')
    ax1.set_title("Weekly Observations & Decoded Hidden Health States", fontweight='bold', pad=10)
    lines = [p1, p2, p3]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # 2. State Transition Probability Matrix Heatmap
    ax2 = axes[0, 1]
    sns.heatmap(hmm.A, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
                xticklabels=hmm.STATE_NAMES, yticklabels=hmm.STATE_NAMES, ax=ax2,
                annot_kws={"size": 12, "weight": "bold"})
    ax2.set_title("Learned State Transition Probability Matrix A", fontweight='bold', pad=10)
    ax2.set_xlabel("To State (t + 1)", fontweight='bold')
    ax2.set_ylabel("From State (t)", fontweight='bold')
    
    # 3. Posterior Probability Distribution Curves
    ax3 = axes[1, 0]
    for k in range(hmm.n_states):
        ax3.plot(weeks, posteriors[:, k], marker="o", linewidth=2.2,
                 color=hmm.STATE_COLORS[k], label=f"P({hmm.STATE_NAMES[k]} | Observations)")
    ax3.set_xlabel("Growth Cycle (Week)", fontweight='bold')
    ax3.set_ylabel("Posterior Probability P(S_t | O)", fontweight='bold')
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Forward-Backward Posterior Probability Curves", fontweight='bold', pad=10)
    ax3.legend(loc="best", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    # 4. Feature Space Clustering by State
    ax4 = axes[1, 1]
    palette = {"Healthy": "#22c55e", "Stressed": "#eab308", "Disease-Suspected": "#ef4444"}
    sns.scatterplot(data=df, x="humidity_pct", y="ndvi", hue="ground_truth_state",
                    palette=palette, s=70, alpha=0.85, ax=ax4, style="ground_truth_state")
    for k in range(hmm.n_states):
        mean_hum = hmm.means[k, 2]
        mean_ndvi = hmm.means[k, 0]
        ax4.scatter(mean_hum, mean_ndvi, color=hmm.STATE_COLORS[k], s=250, marker="X",
                    edgecolors="black", linewidth=1.5, zorder=5, label=f"Centroid: {hmm.STATE_NAMES[k]}")
    ax4.set_title("Emission Feature Space: Humidity vs NDVI", fontweight='bold', pad=10)
    ax4.set_xlabel("Relative Humidity (%)", fontweight='bold')
    ax4.set_ylabel("NDVI", fontweight='bold')
    ax4.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax4.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"[*] Visualizations saved successfully to: '{output_file}'\n")

def main():
    print_header("HIDDEN MARKOV MODEL: CROP-HEALTH STATE DETECTION")
    print("Project: crop-health-hmm")
    print("Hidden States: Healthy (0) | Stressed (1) | Disease-Suspected (2)")
    print("Observations: Weekly sequences of [NDVI, Temp, Humidity, Soil Moisture]\n")
    
    df = connect_db_and_load_data()
    
    feature_cols = ["ndvi", "temperature_c", "humidity_pct", "soil_moisture_pct"]
    state_to_idx = {"Healthy": 0, "Stressed": 1, "Disease-Suspected": 2}
    
    sequences_X = []
    sequences_states = []
    field_ids = df["field_id"].unique()
    
    for fid in field_ids:
        sub_df = df[df["field_id"] == fid].sort_values("week")
        X = sub_df[feature_cols].values
        states = np.array([state_to_idx[s] for s in sub_df["ground_truth_state"].values])
        sequences_X.append(X)
        sequences_states.append(states)
        
    print(f"[*] Prepared {len(sequences_X)} field sequences ({len(df)} total observation weeks).")
    
    hmm = GaussianHMM(n_states=3, feature_names=feature_cols)
    print("\n[*] Initializing and calibrating Gaussian HMM parameters...")
    hmm.fit_supervised(sequences_X, sequences_states)
    hmm.fit_em(sequences_X, n_iter=10)
    print("[*] Model fitting completed successfully.\n")
    
    print("-" * 70)
    print("MODEL PARAMETER MATRIX")
    print("-" * 70)
    print(f"Initial State Probabilities (π):")
    for s_name, p in zip(hmm.STATE_NAMES, hmm.pi):
        print(f"  - {s_name:<20}: {p:.4f} ({p*100:.1f}%)")
        
    print("\nState Transition Probability Matrix A (P(S_t+1 | S_t)):")
    header = f"{'From \\ To':<20}" + "".join([f"{name:>16}" for name in hmm.STATE_NAMES])
    print(header)
    print("-" * len(header))
    for i, from_name in enumerate(hmm.STATE_NAMES):
        row = f"{from_name:<20}" + "".join([f"{hmm.A[i, j]:>16.4f}" for j in range(3)])
        print(row)
        
    print("\nGaussian Emission Centroids (Means μ):")
    feat_header = f"{'State':<20}" + "".join([f"{name:>15}" for name in feature_cols])
    print(feat_header)
    print("-" * len(feat_header))
    for k, s_name in enumerate(hmm.STATE_NAMES):
        row = f"{s_name:<20}" + "".join([f"{hmm.means[k, d]:>15.2f}" for d in range(4)])
        print(row)
    print("-" * 70)
    
    total_obs = 0
    correct_obs = 0
    print("\n[*] Running Viterbi sequence decoding on all field datasets...")
    for fid, X, true_s in zip(field_ids, sequences_X, sequences_states):
        pred_indices, _ = hmm.viterbi(X)
        matches = np.sum(np.array(pred_indices) == true_s)
        correct_obs += matches
        total_obs += len(true_s)
        
    acc = (correct_obs / total_obs) * 100
    print(f"[*] Overall Viterbi Sequence Decoding Concordance: {correct_obs}/{total_obs} ({acc:.2f}%)\n")
    
    sample_fields = ["FIELD-01", "FIELD-03", "FIELD-04"]
    for sf in sample_fields:
        sub = df[df["field_id"] == sf].sort_values("week")
        pred_res = hmm.predict_sequence(sub[feature_cols].values)
        print_ascii_timeline(
            field_id=sf,
            crop_type=sub["crop_type"].iloc[0],
            weeks=sub["week"].values,
            ndvis=sub["ndvi"].values,
            decoded_states=pred_res["decoded_states"],
            true_states=sub["ground_truth_state"].values
        )
        
    output_png = os.path.join(SCRIPT_DIR, "crop_health_hmm_analysis.png")
    generate_visualizations(df, hmm, sample_field="FIELD-04", output_file=output_png)
    
    print_header("CUSTOM WEEKLY SEQUENCE INFERENCE DEMO")
    print("Testing a severe drought scenario sequence (4 weeks):")
    drought_seq = np.array([
        [0.65, 27.0, 60.0, 42.0],
        [0.54, 32.5, 42.0, 24.0],
        [0.46, 36.0, 34.0, 16.0],
        [0.40, 37.5, 30.0, 12.0]
    ])
    d_res = hmm.predict_sequence(drought_seq)
    for w, (row, state) in enumerate(zip(drought_seq, d_res["decoded_states"]), 1):
        post = d_res["posteriors"][w - 1]
        probs_str = f"H: {post[0]:.2f} | S: {post[1]:.2f} | D: {post[2]:.2f}"
        print(f"  Week {w}: NDVI={row[0]:.2f}, Temp={row[1]}°C, Hum={row[2]}%, Moist={row[3]}% -> Decoded: {state:<18} ({probs_str})")

    print("\nTesting a high-humidity fungal disease outbreak sequence (4 weeks):")
    disease_seq = np.array([
        [0.72, 25.0, 75.0, 50.0],
        [0.66, 26.0, 89.0, 62.0],
        [0.48, 25.5, 94.0, 66.0],
        [0.36, 25.0, 92.0, 64.0]
    ])
    dis_res = hmm.predict_sequence(disease_seq)
    for w, (row, state) in enumerate(zip(disease_seq, dis_res["decoded_states"]), 1):
        post = dis_res["posteriors"][w - 1]
        probs_str = f"H: {post[0]:.2f} | S: {post[1]:.2f} | D: {post[2]:.2f}"
        print(f"  Week {w}: NDVI={row[0]:.2f}, Temp={row[1]}°C, Hum={row[2]}%, Moist={row[3]}% -> Decoded: {state:<18} ({probs_str})")
        
    print("\n" + "=" * 70)
    print("Hidden Markov Model Analysis Completed Successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
