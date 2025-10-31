import os
import pandas as pd
import matplotlib.pyplot as plt

# Base directory of your project
base_dir = "."

# Path to summary
summary_path = os.path.join(base_dir, "all_experiment_summary.csv")

if not os.path.exists(summary_path):
    print("⚠️ all_experiment_summary.csv not found. Run summary.py first.")
    exit()

# Read summary and sort
df = pd.read_csv(summary_path)
df["final_val_loss"] = pd.to_numeric(df["final_val_loss"], errors="coerce")
df = df.dropna(subset=["final_val_loss"])
df = df.sort_values(by="final_val_loss", ascending=True)

# Select top 5 experiments
top_experiments = df.head(5)
print("\n🏆 Top 5 Best Experiments by Validation Loss:\n")
print(top_experiments[["experiment", "final_val_loss", "final_train_loss", "n_head", "n_embd", "batch_size", "dropout"]])

# Create folder for plots
plot_dir = os.path.join(base_dir, "plots_top5")
os.makedirs(plot_dir, exist_ok=True)

# Plot each of the top 5 experiments
for _, row in top_experiments.iterrows():
    exp = row["experiment"]
    exp_path = os.path.join(base_dir, exp)
    csv_path = os.path.join(exp_path, "training_log.csv")

    if os.path.exists(csv_path):
        try:
            df_log = pd.read_csv(csv_path)
            plt.figure(figsize=(8, 4))
            plt.plot(df_log["iter"], df_log["train_loss"], "--", label="Train Loss", linewidth=1.5)
            plt.plot(df_log["iter"], df_log["val_loss"], label="Validation Loss", linewidth=1.5)
            plt.title(f"Loss Curve — {exp}", fontsize=12)
            plt.xlabel("Iterations", fontsize=10)
            plt.ylabel("Loss", fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(plot_dir, f"{exp}_loss_curve.png")
            plt.savefig(save_path)
            plt.close()

            print(f"✅ Saved: {save_path}")
        except Exception as e:
            print(f"⚠️ Skipped {exp}: {e}")
    else:
        print(f"⚠️ No training_log.csv found for {exp}")

# --- Combine all top-5 validation curves into a single comparison plot ---
plt.figure(figsize=(10, 6))
for _, row in top_experiments.iterrows():
    exp = row["experiment"]
    csv_path = os.path.join(base_dir, exp, "training_log.csv")
    if os.path.exists(csv_path):
        df_log = pd.read_csv(csv_path)
        plt.plot(df_log["iter"], df_log["val_loss"], label=f"{exp} (val)", linewidth=1.5)

plt.title("Top 5 Experiments — Validation Loss Comparison", fontsize=14)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Validation Loss", fontsize=12)
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

combined_path = os.path.join(plot_dir, "top5_validation_comparison.png")
plt.savefig(combined_path)
plt.show()

print(f"\n📊 Combined plot saved: {combined_path}")
