import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "C:/Users/ganes/Downloads/nanoGPT-master/nanoGPT-master"
exp_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("out_")]

# Folder to store plots
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# --- 1️⃣ Individual Experiment Plots ---
for exp in exp_dirs:
    csv_path = os.path.join(exp, "training_log.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        exp_name = os.path.basename(exp)

        plt.figure(figsize=(8, 4))
        plt.plot(df["iter"], df["train_loss"], "--", label="Train Loss", linewidth=1.5)
        plt.plot(df["iter"], df["val_loss"], label="Validation Loss", linewidth=1.5)
        plt.title(f"Loss Curve: {exp_name}", fontsize=12)
        plt.xlabel("Iterations", fontsize=10)
        plt.ylabel("Loss", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(plot_dir, f"{exp_name}_loss_curve.png")
        plt.savefig(save_path)
        plt.close()

        print(f"✅ Saved: {save_path}")

print("\nAll individual plots saved successfully.\n")

# --- 2️⃣ Group Comparison by Key Hyperparameters ---
import re

summary_data = []
for exp in exp_dirs:
    name = os.path.basename(exp)
    match = re.search(r"nh(\d+)_ne(\d+)_b(\d+)_mi(\d+)_do([0-9.]+)", name)
    if match:
        n_head, n_embd, batch, max_iter, dropout = match.groups()
        metrics_path = os.path.join(exp, "final_metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            val_loss = df["final_val_loss"].iloc[0]
            summary_data.append({
                "experiment": name,
                "n_head": int(n_head),
                "n_embd": int(n_embd),
                "batch_size": int(batch),
                "max_iters": int(max_iter),
                "dropout": float(dropout),
                "val_loss": val_loss
            })

if summary_data:
    summary_df = pd.DataFrame(summary_data)

    # --- Group 1: Embedding Size vs Validation Loss ---
    plt.figure(figsize=(6, 4))
    for nh in sorted(summary_df["n_head"].unique()):
        sub_df = summary_df[summary_df["n_head"] == nh]
        plt.plot(sub_df["n_embd"], sub_df["val_loss"], marker="o", label=f"n_head={nh}")
    plt.title("Validation Loss vs Embedding Size", fontsize=12)
    plt.xlabel("Embedding Size (n_embd)")
    plt.ylabel("Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "val_loss_vs_embd.png"))
    plt.close()

    # --- Group 2: Dropout vs Validation Loss ---
    plt.figure(figsize=(6, 4))
    for ne in sorted(summary_df["n_embd"].unique()):
        sub_df = summary_df[summary_df["n_embd"] == ne]
        plt.plot(sub_df["dropout"], sub_df["val_loss"], marker="o", label=f"n_embd={ne}")
    plt.title("Validation Loss vs Dropout", fontsize=12)
    plt.xlabel("Dropout")
    plt.ylabel("Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "val_loss_vs_dropout.png"))
    plt.close()

    # --- Group 3: Batch Size vs Validation Loss ---
    plt.figure(figsize=(6, 4))
    for ne in sorted(summary_df["n_embd"].unique()):
        sub_df = summary_df[summary_df["n_embd"] == ne]
        plt.plot(sub_df["batch_size"], sub_df["val_loss"], marker="o", label=f"n_embd={ne}")
    plt.title("Validation Loss vs Batch Size", fontsize=12)
    plt.xlabel("Batch Size")
    plt.ylabel("Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "val_loss_vs_batch.png"))
    plt.close()

    print("📊 Saved grouped comparison plots successfully.")
else:
    print("⚠️ No valid experiments found to summarize.")
