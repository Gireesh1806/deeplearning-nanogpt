
import os
import itertools

"""
NanoGPT Hyperparameter Experiments
====================================
This script runs systematic experiments varying key hyperparameters.

Assignment Requirements:
- Total: 2^7 = 128 combinations
- Each group member runs 32 experiments (2^5 = 32)

Group Member Assignments:
- Member 1: [block_size=64,  n_layer=4, *, *, *, *, *]
- Member 2: [block_size=64,  n_layer=6, *, *, *, *, *]
- Member 3: [block_size=128, n_layer=4, *, *, *, *, *]
- Member 4: [block_size=128, n_layer=6, *, *, *, *, *]

Varying Parameters (for all members):
- n_head: [4, 8]
- n_embd: [128, 256] (must be divisible by n_head)
- batch_size: [8, 16]
- max_iters: [1000, 2000]
- dropout: [0.1, 0.2]
"""

# =============================================================================
# CONFIGURATION - SET YOUR GROUP MEMBER NUMBER HERE
# =============================================================================
GROUP_MEMBER = 1  # Change this to 1, 2, 3, or 4

# =============================================================================
# GROUP MEMBER SPECIFIC SETTINGS
# =============================================================================
if GROUP_MEMBER == 1:
    block_size = 64
    n_layer = 4
elif GROUP_MEMBER == 2:
    block_size = 64
    n_layer = 6
elif GROUP_MEMBER == 3:
    block_size = 128
    n_layer = 4
elif GROUP_MEMBER == 4:
    block_size = 128
    n_layer = 6
else:
    raise ValueError("GROUP_MEMBER must be 1, 2, 3, or 4")

# =============================================================================
# VARYING HYPERPARAMETERS (Same for all group members)
# =============================================================================
n_heads = [4, 8]
n_embds = [128, 256]
batch_sizes = [8, 16]
max_iters_list = [1000, 2000]
dropouts = [0.1, 0.2]

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================
print("=" * 80)
print(f"STARTING EXPERIMENTS FOR GROUP MEMBER {GROUP_MEMBER}")
print("=" * 80)
print(f"Fixed Parameters:")
print(f"  - block_size: {block_size}")
print(f"  - n_layer: {n_layer}")
print(f"\nVarying Parameters:")
print(f"  - n_head: {n_heads}")
print(f"  - n_embd: {n_embds}")
print(f"  - batch_size: {batch_sizes}")
print(f"  - max_iters: {max_iters_list}")
print(f"  - dropout: {dropouts}")
print(f"\nTotal Experiments: {len(n_heads) * len(n_embds) * len(batch_sizes) * len(max_iters_list) * len(dropouts)}")
print("=" * 80)

# Counter for experiments
experiment_count = 0

# Iterate over all combinations
for n_head, n_embd, batch_size, max_iters, dropout in itertools.product(
    n_heads, n_embds, batch_sizes, max_iters_list, dropouts
):
    experiment_count += 1
    
    # Create descriptive output directory name
    out_dir = f"out_bs{block_size}_nl{n_layer}_nh{n_head}_ne{n_embd}_b{batch_size}_mi{max_iters}_do{dropout}"
    
    # Build the training command
    cmd = (
        f"python train.py config/train_shakespeare_char.py "
        f"--block_size={block_size} "
        f"--n_layer={n_layer} "
        f"--n_head={n_head} "
        f"--n_embd={n_embd} "
        f"--batch_size={batch_size} "
        f"--max_iters={max_iters} "
        f"--dropout={dropout} "
        f"--out_dir={out_dir}"
    )
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {experiment_count}/32 - Group Member {GROUP_MEMBER}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  block_size  = {block_size}")
    print(f"  n_layer     = {n_layer}")
    print(f"  n_head      = {n_head}")
    print(f"  n_embd      = {n_embd}")
    print(f"  batch_size  = {batch_size}")
    print(f"  max_iters   = {max_iters}")
    print(f"  dropout     = {dropout}")
    print(f"  out_dir     = {out_dir}")
    print(f"\nCommand: {cmd}")
    print(f"{'='*80}\n")
    
    # Execute the training command
    os.system(cmd)

print("\n" + "=" * 80)
print(f"ALL {experiment_count} EXPERIMENTS COMPLETED FOR GROUP MEMBER {GROUP_MEMBER}!")
print("=" * 80)
print("\nNext Steps:")
print("1. Run evaluate.py to extract metrics from training logs")
print("2. Run summary.py to create a summary CSV of all experiments")
print("3. Run top.py to identify the best performing models")
print("4. Run plot.py to visualize loss curves")
print("5. Run run_sampling.py to generate text samples from trained models")
