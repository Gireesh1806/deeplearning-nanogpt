import os
import pandas as pd
import re

"""
Aggregate all experiment results into a single summary CSV
Creates: all_experiment_summary.csv
This version is robust to different CSV column names
"""

base_dir = "."

# Find all experiment output directories
exp_dirs = [d for d in os.listdir(base_dir) if d.startswith("out_bs") and os.path.isdir(d)]

print(f"\n{'='*80}")
print(f"CREATING EXPERIMENT SUMMARY")
print(f"{'='*80}\n")

summary_data = []
missing_metrics = []

for exp_dir in sorted(exp_dirs):
    exp_path = os.path.join(base_dir, exp_dir)
    metrics_path = os.path.join(exp_path, "final_metrics.csv")
    
    if not os.path.exists(metrics_path):
        missing_metrics.append(exp_dir)
        continue
    
    try:
        # Read final metrics
        df_metrics = pd.read_csv(metrics_path)
        
        if len(df_metrics) > 0:
            row = df_metrics.iloc[0].to_dict()
            
            # Extract hyperparameters from directory name
            # Format: out_bs{block_size}_nl{n_layer}_nh{n_head}_ne{n_embd}_b{batch_size}_mi{max_iters}_do{dropout}
            match = re.search(r"bs(\d+)_nl(\d+)_nh(\d+)_ne(\d+)_b(\d+)_mi(\d+)_do([0-9.]+)", exp_dir)
            
            if match:
                block_size, n_layer, n_head, n_embd, batch_size, max_iters, dropout = match.groups()
                
                # Create a clean row with consistent column names
                clean_row = {
                    'experiment': exp_dir,
                    'block_size': int(block_size),
                    'n_layer': int(n_layer),
                    'n_head': int(n_head),
                    'n_embd': int(n_embd),
                    'batch_size': int(batch_size),
                    'max_iters': int(max_iters),
                    'dropout': float(dropout)
                }
                
                # Extract loss values - handle different possible column names
                # Try various possible names for training loss
                train_loss_cols = ['final_train_loss', 'train_loss', 'training_loss', 'final_training_loss']
                for col in train_loss_cols:
                    if col in row:
                        clean_row['final_train_loss'] = float(row[col])
                        break
                
                # Try various possible names for validation loss
                val_loss_cols = ['final_val_loss', 'val_loss', 'validation_loss', 'final_validation_loss']
                for col in val_loss_cols:
                    if col in row:
                        clean_row['final_val_loss'] = float(row[col])
                        break
                
                # If we couldn't find losses in metrics file, try to get from training log
                if 'final_train_loss' not in clean_row or 'final_val_loss' not in clean_row:
                    log_path = os.path.join(exp_path, "training_log.csv")
                    if os.path.exists(log_path):
                        try:
                            df_log = pd.read_csv(log_path)
                            if len(df_log) > 0:
                                clean_row['final_train_loss'] = float(df_log['train_loss'].iloc[-1])
                                clean_row['final_val_loss'] = float(df_log['val_loss'].iloc[-1])
                        except:
                            pass
                
                # Add any other columns from the original metrics
                for key, value in row.items():
                    if key not in clean_row and key != 'Unnamed: 0':
                        clean_row[key] = value
                
                summary_data.append(clean_row)
                
                # Print status
                if 'final_val_loss' in clean_row:
                    print(f"✓ {exp_dir} (val_loss={clean_row['final_val_loss']:.4f})")
                else:
                    print(f"⚠️  {exp_dir} (no loss values found)")
            else:
                print(f"⚠️  {exp_dir}: Could not parse hyperparameters from name")
                
    except Exception as e:
        print(f"❌ {exp_dir}: Error - {e}")

# Create summary DataFrame
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    
    # Ensure we have the essential columns
    required_cols = ['experiment', 'final_train_loss', 'final_val_loss']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    
    if missing_cols:
        print(f"\n⚠️  WARNING: Missing columns in summary: {missing_cols}")
        print("Some experiments may not have complete data.")
    
    # Reorder columns for better readability
    column_order = [
        'experiment',
        'final_train_loss',
        'final_val_loss',
        'block_size',
        'n_layer',
        'n_head',
        'n_embd',
        'batch_size',
        'max_iters',
        'dropout'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in summary_df.columns]
    
    # Add any additional columns
    for col in summary_df.columns:
        if col not in column_order:
            column_order.append(col)
    
    summary_df = summary_df[column_order]
    
    # Sort by validation loss (best first) if we have that column
    if 'final_val_loss' in summary_df.columns:
        summary_df = summary_df.sort_values('final_val_loss', ascending=True)
    
    # Save to CSV
    output_path = os.path.join(base_dir, "all_experiment_summary.csv")
    summary_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY CREATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"✅ Total experiments: {len(summary_df)}")
    print(f"📊 Saved to: {output_path}")
    
    # Print top 5 models if we have validation loss
    if 'final_val_loss' in summary_df.columns:
        print(f"\n{'='*80}")
        print(f"🏆 TOP 5 MODELS BY VALIDATION LOSS")
        print(f"{'='*80}\n")
        
        top5 = summary_df.head(5)
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"Rank {idx}: {row['experiment']}")
            print(f"  Val Loss:    {row['final_val_loss']:.4f}")
            if 'final_train_loss' in row:
                print(f"  Train Loss:  {row['final_train_loss']:.4f}")
            print(f"  Architecture: n_layer={int(row['n_layer'])}, n_head={int(row['n_head'])}, n_embd={int(row['n_embd'])}")
            print(f"  Training:     batch={int(row['batch_size'])}, iters={int(row['max_iters'])}, dropout={row['dropout']}")
            print()
        
        # Print some statistics
        print(f"{'='*80}")
        print(f"📈 SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Best Val Loss:    {summary_df['final_val_loss'].min():.4f}")
        print(f"Worst Val Loss:   {summary_df['final_val_loss'].max():.4f}")
        print(f"Mean Val Loss:    {summary_df['final_val_loss'].mean():.4f}")
        print(f"Median Val Loss:  {summary_df['final_val_loss'].median():.4f}")
        print(f"Std Dev Val Loss: {summary_df['final_val_loss'].std():.4f}")
        
        # Analysis by hyperparameter
        print(f"\n{'='*80}")
        print(f"📊 MEAN VALIDATION LOSS BY HYPERPARAMETER")
        print(f"{'='*80}")
        
        for param in ['n_layer', 'n_head', 'n_embd', 'batch_size', 'max_iters', 'dropout']:
            if param in summary_df.columns:
                grouped = summary_df.groupby(param)['final_val_loss'].mean().sort_values()
                print(f"\n{param}:")
                for value, loss in grouped.items():
                    print(f"  {param}={value}: {loss:.4f}")
    
else:
    print("❌ No experiment data found!")

if missing_metrics:
    print(f"\n{'='*80}")
    print(f"⚠️  WARNING: {len(missing_metrics)} experiments missing final_metrics.csv")
    print(f"{'='*80}")
    print("Run 'python evaluate.py' first to extract metrics from training logs")
    for exp in missing_metrics[:5]:  # Show first 5
        print(f"  - {exp}")
    if len(missing_metrics) > 5:
        print(f"  ... and {len(missing_metrics) - 5} more")

print(f"\nNext steps:")
print("1. Run 'python top.py' to visualize top 5 models")
print("2. Run 'python plot.py' to create all loss curve plots")
print("3. Run 'python run_sampling.py' to generate text samples")