import os
import pandas as pd

base_dir = "."
exp_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("out_")]

results = []
for exp in exp_dirs:
    csv_path = os.path.join(exp, "final_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["experiment"] = os.path.basename(exp)
        results.append(df)

if results:
    all_results = pd.concat(results, ignore_index=True)
    all_results = all_results.sort_values(by="final_val_loss", ascending=True)

    print("\n🏆 Final Experiment Comparison:")
    print(all_results[[
        "experiment", "final_val_loss", "final_train_loss",
        "max_iters", "n_layer", "n_head", "n_embd", "batch_size", "dropout"
    ]].to_string(index=False))

    # Save summary
    all_results.to_csv(os.path.join(base_dir, "all_experiment_summary.csv"), index=False)
    print(f"\n✅ Saved all results to: {os.path.join(base_dir, 'all_experiment_summary.csv')}")
else:
    print("⚠️ No final_metrics.csv files found!")
