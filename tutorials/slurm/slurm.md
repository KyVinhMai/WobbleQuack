
# **📄 General SLURM sbatch Script Template**

```bash
#!/bin/bash
#SBATCH --job-name=llm_experiment           # Job name (appears in queue)
#SBATCH --partition=gpu                     # Partition: gpu, free-gpu, standard, free
#SBATCH --account=YOUR_LAB_GPU              # Your lab's GPU account (required for gpu partition)
#SBATCH --gres=gpu:A30:1                   # GPU type and count (V100:1, A30:1, A100:1, L40S:1)
#SBATCH --cpus-per-task=8                   # CPU cores (typically 4-8 per GPU)
#SBATCH --mem=64G                           # Memory request (be realistic!)
#SBATCH --time=12:00:00                     # Wall time (HH:MM:SS) - max varies by partition
#SBATCH --nodes=1                           # Almost always 1 for LLM work
#SBATCH --tmp=50G                           # Request scratch space for temp files
#SBATCH --constraint=fastscratch            # Request nodes with fast NVMe scratch
#SBATCH --output=logs/job_%j.out            # Output log (%j = job ID)
#SBATCH --error=logs/job_%j.err             # Error log
#SBATCH --mail-type=END,FAIL                # Email notifications (optional: Really bad for work life balance)
#SBATCH [--mail-user=your-email@uci.edu](mailto:--mail-user=your-email@uci.edu)      # Your email (optional)

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (ALWAYS specify versions!)
module load python/3.10.2
module load cuda/11.7.1

# Activate conda environment
source ~/.bashrc
conda activate llm_experiments

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set Hugging Face token and cache to pub area (not scratch)
export HF_TOKEN="your_huggingface_token_here"
export TRANSFORMERS_CACHE="/pub/$USER/hf_cache"
mkdir -p $TRANSFORMERS_CACHE

# Set up results directory in permanent storage
RESULTS_DIR="/pub/$USER/results/experiment_$SLURM_JOB_ID"
mkdir -p $RESULTS_DIR

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Scratch dir: $TMPDIR"
echo "Results dir: $RESULTS_DIR"
echo "Start time: $(date)"

# Copy input files to scratch directory ($TMPDIR is automatically created)
echo "Copying input files to scratch..."
cd $TMPDIR

# Copy your code and data files to scratch for faster I/O
cp $SLURM_SUBMIT_DIR/run_experiments.py .
cp $SLURM_SUBMIT_DIR/data_for_LLM_testing.csv .
cp -r $SLURM_SUBMIT_DIR/utils/ .

# Add current directory to Python path
export PYTHONPATH=$TMPDIR:$PYTHONPATH

# Function to sync results back to permanent storage
sync_results() {
    echo "Syncing results to permanent storage..."
    if [ -d "$TMPDIR/results" ]; then
        rsync -av --update $TMPDIR/results/ $RESULTS_DIR/
        sync  # Ensure filesystem is synced
        echo "Results synced at $(date)"
    else
        echo "Warning: No results directory found in scratch"
    fi
}

# Set up trap to sync results on job termination (normal or early)
trap sync_results EXIT

# Run your experiment (working from scratch directory)
echo "Starting experiment..."
python run_experiments.py \
    --models "7" \
    --prompt "simplified_prompt" \
    --temperatures "0.1,0.5,1.0" \
    --scale_range "1.0" \
    --results_dir "results" \
    --hf_token "$HF_TOKEN"

# Sync results one more time at the end
sync_results

echo "Job completed at: $(date)"

# Optional: Clean up scratch (happens automatically, but explicit is good)
rm -rf $TMPDIR/*

```

---

# **🔧 Key Corrections & Best Practices**

## **Scratch Storage Usage**

```bash
#SBATCH --tmp=50G                    # Request adequate scratch space
#SBATCH --constraint=fastscratch     # Use NVMe nodes for better I/O

# Always work from $TMPDIR for temporary files
cd $TMPDIR

# Copy inputs TO scratch at start
cp $SLURM_SUBMIT_DIR/input_files $TMPDIR/

# Copy results FROM scratch before job ends
rsync -av $TMPDIR/results/ /pub/$USER/permanent_results/

```

## **Different Scratch Usage Patterns**

### **Pattern 1: Small temporary files (your case)**

```bash
# Your experiment creates model outputs, logs, CSVs
#SBATCH --tmp=20G

# Work in scratch, sync results periodically
cd $TMPDIR
# ... run experiments ...
rsync -av results/ /pub/$USER/final_results/

```

### **Pattern 2: Large model caching**

```bash
# If you need to cache large models temporarily
#SBATCH --tmp=100G
#SBATCH --constraint=fastscratch

# Cache models in scratch for speed
export TRANSFORMERS_CACHE=$TMPDIR/hf_cache
# ... your code automatically downloads there ...
# Sync important outputs back to permanent storage

```

### **Pattern 3: Preprocessing large datasets**

```bash
# Processing large datasets with many small I/O operations
#SBATCH --tmp=200G
#SBATCH --constraint=fastscratch

# Copy dataset to scratch
cp /pub/$USER/large_dataset.csv $TMPDIR/
cd $TMPDIR

# Process with many small reads/writes (fast on local NVMe)
python preprocess.py --input large_dataset.csv --output processed/
# Copy final results back
cp -r processed/ /pub/$USER/

```

---

# **📁 File Organization Best Practices**

```bash
# Your permanent storage structure
/pub/$USER/
├── code/                    # Your scripts (backed up)
├── data/                    # Input datasets
├── models/                  # Fine-tuned models
├── results/                 # Experiment outputs
│   └── experiment_123456/   # Organized by job ID
└── hf_cache/               # HuggingFace model cache

# During job execution in $TMPDIR
$TMPDIR/
├── run_experiments.py      # Copied from submit dir
├── data_for_LLM_testing.csv
├── utils/                  # Your utility modules
└── results/                # Generated during run
    └── model_7/            # Synced back to /pub

```

---

# **⚠️ Critical Reminders**

1. **Never store important data only in $TMPDIR** - it's deleted when job ends
2. **Always copy input files TO scratch** - don't read from /pub during compute
3. **Use rsync with --update flag** - avoids overwriting newer files
4. **Set up trap for early termination** - ensures results are saved even if job is cancelled
5. **Request appropriate scratch space** - better to overestimate than run out
6. **Use fastscratch constraint** - NVMe is much faster than regular disks

This approach will give you much better I/O performance and follow HPC3 best practices!

## **🎯 Core SLURM Commands**

### **Job Submission & Management**

```bash
# Submit a job
sbatch my_script.sh                        # Submit batch job
sbatch --wrap="python train.py"            # Quick one-liner job

# Monitor jobs
squeue -u $USER                            # Your jobs only
squeue -p gpu                              # All GPU partition jobs
scontrol show job [job_id]                 # Detailed job info

# Cancel jobs
scancel [job_id]                           # Cancel specific job
scancel -u $USER                           # Cancel ALL your jobs (careful!)
scancel -u $USER -p gpu                    # Cancel your GPU jobs only

# Interactive sessions
srun -p free-gpu --gres=gpu:V100:1 --mem=32G --time=2:00:00 --pty /bin/bash -i

```

### **Job Status & Efficiency**

```bash
# Check job efficiency after completion
seff [job_id]                              # CPU/memory efficiency
sacct -j [job_id] --format=JobID,JobName,MaxRSS,Elapsed  # Accounting info

# Check your account balance
sbank balance statement -u $USER           # All accounts
sbank balance statement -a YOUR_LAB_GPU    # Specific account

```

---

## **🏗️ Partition Guide**

| Partition | Cost | GPUs | Max Time | When to Use |
| --- | --- | --- | --- | --- |
| `free-gpu` | Free | ✅ | 3 days | Testing, debugging, small experiments |
| `gpu` | Paid | ✅ | 14 days | Production runs, important experiments |
| `free` | Free | ❌ | 3 days | CPU-only preprocessing |
| `standard` | Paid | ❌ | 14 days | CPU-intensive work |

---

## **💾 Resource Guidelines**

### **GPU Selection**

```bash
# Choose based on your model size and memory needs:
--gres=gpu:V100:1     # 16GB VRAM - Good for 1B, 3B, 7B models
--gres=gpu:A30:1      # 24GB VRAM - Better for < 7B, can handle some 13B
--gres=gpu:A100:1     # 80GB VRAM - Best for large models (13B+), Need DPP For 70B+
--gres=gpu:L40S:1     # 48GB VRAM - Decent for most work, mainly for Computer Vison

```

### **Memory & CPU Guidelines**

```bash
# Conservative estimates:
# 7B model:  --mem=32G --cpus-per-task=4
# 13B model: --mem=64G --cpus-per-task=8
# 70B model: --mem=128G --cpus-per-task=16 (if using multiple GPUs)

# Rule of thumb: 4-8 CPUs per GPU, 8-16GB RAM per GPU
```

### **Time Limits**

```bash
# Be realistic but generous:
--time=01:00:00       # 1 hour - quick tests
--time=06:00:00       # 6 hours - medium experiments
--time=24:00:00       # 1 day - long training runs
--time=72:00:00       # 3 days - maximum for free partitions
```

---

## **🔧 Common Job Patterns**

### **Quick Interactive Testing**

```bash
# Get GPU node for 2 hours
srun -p free-gpu --gres=gpu:V100:1 --mem=32G --time=2:00:00 --pty /bin/bash -i

# Load environment and test
module load python/3.10.2 cuda/11.7.1
conda activate llm_experiments
python -c "import torch; print(torch.cuda.is_available())"
```

### **Array Jobs (Multiple Experiments)**

```bash
#SBATCH --array=1-10                       # Run 10 variations
#SBATCH --output=logs/job_%A_%a.out        # %A=array job ID, %a=task ID

# In your script:
TEMP_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
TEMPERATURE=${TEMP_VALUES[$SLURM_ARRAY_TASK_ID-1]}
python train.py --temperature $TEMPERATURE

```

### **Dependency Jobs (Pipeline)**

```bash
# Submit job 1
JOB1=$(sbatch --parsable preprocess.sh)

# Submit job 2 that waits for job 1
JOB2=$(sbatch --dependency=afterok:$JOB1 --parsable train.sh)

# Submit job 3 that waits for job 2
sbatch --dependency=afterok:$JOB2 evaluate.sh

```

---

## **📊 Monitoring & Debugging**

### **While Job is Running**

```bash
# Watch your job in the queue
watch -n 30 'squeue -u $USER'

# Attach to running job (for debugging)
srun --pty --jobid=[job_id] --overlap /bin/bash

# Check live resource usage
ssh [node_name]  # From squeue output
htop             # CPU/memory usage
nvidia-smi       # GPU usage

```

### **After Job Completes**

```bash
# Check efficiency
seff [job_id]

# View logs
tail logs/job_[job_id].out
tail logs/job_[job_id].err

# Check what went wrong (if failed)
sacct -j [job_id] --format=JobID,State,ExitCode,DerivedExitCode

```

---

## **⚠️ Common Gotchas**

1. **Wrong Partition**: Using `gpu` without a GPU account → Use `free-gpu` for testing
2. **Module Loading**: Always load modules in job script, not `.bashrc`
3. **Path Issues**: Use absolute paths; working directory might not be what you expect
4. **Memory Limits**: Job gets killed if you exceed requested memory
5. **Time Limits**: Job gets killed at time limit regardless of progress
6. **File Permissions**: Make sure scripts are executable (`chmod +x script.sh`)

---

## **🎯 Pro Tips**

- **Start small**: Test with `free-gpu` before using paid hours
- **Save frequently**: Implement checkpointing in long jobs
- **Use scratch**: Copy large temporary files to `$TMPDIR` for faster I/O