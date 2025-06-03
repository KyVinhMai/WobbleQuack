#!/bin/bash
#SBATCH --job-name=llm_experiment           # Job name (appears in queue)
#SBATCH --partition=gpu                     # Partition: gpu, free-gpu, standard, free
#SBATCH --account=YOUR_LAB_GPU              # Your lab's GPU account (required for gpu partition)
#SBATCH --gres=gpu:A30:1                   # GPU type and count (V100:1, A30:1, A100:1, L40S:1)
#SBATCH --cpus-per-task=8                   # CPU cores (typically 4-8 per GPU)
#SBATCH --mem=64G                           # Memory request (be realistic!)
#SBATCH --time=12:00:00                     # Wall time (HH:MM:SS) - max varies by partition
#SBATCH --nodes=1                           # Almost always 1 for LLM work
#SBATCH --output=logs/job_%j.out            # Output log (%j = job ID)
#SBATCH --error=logs/job_%j.err             # Error log
#SBATCH --mail-type=END,FAIL                # Email notifications (optional: Really bad for work life balance) 
#SBATCH --mail-user=your-email@uci.edu      # Your email (optional)

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (ALWAYS specify versions!)
module load python/3.10.2
module load cuda/11.7.1

# Activate your conda environment
source ~/.bashrc
conda activate llm_experiments

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512


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
# Your actual work here
python your_script.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --output_dir "/pub/$USER/results/experiment_$SLURM_JOB_ID" \
    --batch_size 4 \
    --max_length 2048

# Optional: Copy results from scratch to permanent storage
if [ -d "$TMPDIR_CUSTOM" ]; then
    echo "Copying results from scratch..."
    rsync -av $TMPDIR_CUSTOM/ /pub/$USER/results/job_$SLURM_JOB_ID/
fi

echo "Job completed at: $(date)"
