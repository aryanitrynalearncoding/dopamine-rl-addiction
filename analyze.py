# analyze.py
import csv, os
import numpy as np
import matplotlib.pyplot as plt

RESULTS = "results"

def load(cid):
    train, test = [], []
    path = os.path.join(RESULTS, f"cond_{cid}.csv")
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rec = (int(row["episode"]), float(row["notif_rate"]),
                   float(row["task_rate"]), float(row["mean_abs_delta"]))
            (train if row["phase"] == "train" else test).append(rec)
    return train, test

def smooth(x, k=50):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k)/k, mode="valid")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
conditions = "ABCDEFG"
colors = {'A':'gray', 'B':'blue', 'C':'green', 'D':'orange', 'E':'red', 'F':'purple', 'G':'brown'}

for cid in conditions:
    tr, te = load(cid)
    if not tr: continue
    
    axes[0].plot(smooth([r[1] for r in tr]), label=cid, color=colors[cid], alpha=0.8)
    axes[1].plot([r[1] for r in te], label=cid, color=colors[cid], alpha=0.8)
    axes[2].plot(smooth([r[3] for r in tr]), label=cid, color=colors[cid], alpha=0.8)

axes[0].set_title("Train: Notification Rate (Learning)")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Proportion of Actions")

axes[1].set_title("Test: Notification Rate (Extinction Probe)")
axes[1].set_xlabel("Episode")
axes[1].axvline(20, ls="--", c="k", label="End of Probe")

axes[2].set_title("Train: Loss (Proxy for RPE Volatility)")
axes[2].set_xlabel("Episode")

for ax in axes: 
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "summary.png"), dpi=150) # Increased resolution
print("Graph saved to results/summary.png!")

# Print the final Extinction Resistance Table
print("\n--- FINAL EXTINCTION RESISTANCE TABLE ---")
print("Condition | Beta | Intervention | Notif Rate (First 20 Test Episodes)")
print("-" * 70)
for cid in conditions:
    _, te = load(cid)
    if not te: continue
    # Get beta and intervention from the first test row
    with open(os.path.join(RESULTS, f"cond_{cid}.csv")) as f:
        first_test_row = next(r for r in csv.DictReader(f) if r["phase"] == "test")
    beta = first_test_row.get("intervention", "None") # We logged intervention here
    rate = np.mean([r[1] for r in te[:20]])
    print(f"    {cid}     |  {'0.0' if cid in 'ABC' else '0.3'} | {first_test_row['intervention']:15} | {rate:.3f}")
print("-" * 70)