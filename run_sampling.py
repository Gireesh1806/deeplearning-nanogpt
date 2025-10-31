import os
import subprocess

# Base directory where your experiment folders are
BASE_DIR = "."
# Folder to store generated samples
OUTPUT_DIR = "samples"

# Make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all experiment output folders (trained checkpoints)
experiment_dirs = [d for d in os.listdir(BASE_DIR) if d.startswith("out_bs") and os.path.isdir(d)]

# Text prompt to start generation
START_PROMPT = "To be or not to be"
TEMPERATURE = 0.8
TOP_K = 50

for exp_dir in sorted(experiment_dirs):
    print(f"\n🎭 Generating sample for: {exp_dir}\n")

    output_file = os.path.join(OUTPUT_DIR, f"{exp_dir}.txt")

    # Build the command to run sample.py
    cmd = [
        "python",
        "sample.py",
        f"--out_dir={exp_dir}",
        f"--start={START_PROMPT}",
        f"--temperature={TEMPERATURE}",
        f"--top_k={TOP_K}"
    ]

    # Run the command and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stdout, _ = process.communicate()

    # Save both the printed text and model info into a .txt file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(stdout)

    print(f"✅ Sample saved to {output_file}")
