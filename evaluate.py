import os
import pandas as pd
import re

"""
Extract metrics from all experiment training logs
Creates a final_metrics.csv file for each experiment if it doesn't exist
"""

base_dir = "."

# Find all experiment output directories
exp_dirs = [d for d in os.listdir(base_dir) if d.startswith("out_bs") and os.path.isdir(d)]

print(f"\n{'='*80}")
print(f"EVALUATING {len(exp_dirs)} EXPERIMENTS")
print(f"{'='*80}\n")

processed = 0
skipped = 0

for exp_dir in sorted(exp_dirs):
    exp_path = os.path.join(base_dir, exp_dir)
    
    # Check if final_metrics.csv already exists
    metrics_path = os.path.join(exp_path, "final_metrics.csv")
    if os.path.exists(metrics_path):
        print(f"✓ {exp_dir}: final_metrics.csv exists")
        skipped += 1
        continue
    
    # Try to extract metrics from training_log.csv
    log_path = os.path.join(exp_path, "training_log.csv")
    if not os.path.exists(log_path):
        print(f"⚠️  {exp_dir}: No training_log.csv found - skipping")
        continue
    
    try:
        # Read training log
        df_log = pd.read_csv(log_path)
        
        if len(df_log) == 0:
            print(f"⚠️  {exp_dir}: Empty training log - skipping")
            continue
        
        # Helper function to parse tensor strings
        def parse_loss_value(value):
            """Parse loss value which might be a float, tensor string, or other format"""
            if pd.isna(value):
                return None
            
            value_str = str(value).strip()
            
            # Handle tensor format: 'tensor(1.2345)'
            if value_str.startswith('tensor(') and value_str.endswith(')'):
                number_str = value_str[7:-1]  # Extract number from 'tensor(...)'
                return float(number_str)
            
            # Handle regular float
            try:
                return float(value_str)
            except ValueError:
                return None
        
        # Get final losses with proper parsing
        final_train_loss = parse_loss_value(df_log['train_loss'].iloc[-1])
        final_val_loss = parse_loss_value(df_log['val_loss'].iloc[-1])
        max_iter = int(df_log['iter'].iloc[-1])
        
        if final_train_loss is None or final_val_loss is None:
            print(f"⚠️  {exp_dir}: Could not parse loss values - skipping")
            continue
        
        # Extract hyperparameters from directory name
        # Format: out_bs{block_size}_nl{n_layer}_nh{n_head}_ne{n_embd}_b{batch_size}_mi{max_iters}_do{dropout}
        match = re.search(r"bs(\d+)_nl(\d+)_nh(\d+)_ne(\d+)_b(\d+)_mi(\d+)_do([0-9.]+)", exp_dir)
        
        if match:
            block_size, n_layer, n_head, n_embd, batch_size, max_iters, dropout = match.groups()
            
            # Create final_metrics.csv
            metrics_df = pd.DataFrame([{
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'max_iters': int(max_iters),
                'n_layer': int(n_layer),
                'n_head': int(n_head),
                'n_embd': int(n_embd),
                'batch_size': int(batch_size),
                'dropout': float(dropout),
                'block_size': int(block_size),
                'actual_iters': int(max_iter)
            }])
            
            metrics_df.to_csv(metrics_path, index=False)
            print(f"✅ {exp_dir}: Created final_metrics.csv (val_loss={final_val_loss:.4f})")
            processed += 1
        else:
            print(f"⚠️  {exp_dir}: Could not parse hyperparameters from directory name")
            
    except Exception as e:
        print(f"❌ {exp_dir}: Error - {e}")

print(f"\n{'='*80}")
print(f"EVALUATION COMPLETE")
print(f"{'='*80}")
print(f"✅ Processed: {processed} experiments")
print(f"⏭️  Skipped (already exist): {skipped} experiments")
print(f"📊 Total: {processed + skipped} experiments with metrics")
print(f"\nNext step: Run 'python summary.py' to create all_experiment_summary.csv")