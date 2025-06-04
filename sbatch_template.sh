#!/bin/bash
#SBATCH --job-name=llm_experiment           # Job name (appears in queue)
#SBATCH --partition=gpu                     # Partition: gpu, free-gpu, standard, free
#SBATCH --account=YOUR_LAB_GPU              # Your lab's GPU account (required for gpu partition)
#SBATCH --gres=gpu:V100:1                   # GPU type and count (V100:1, A30:1, A100:1, L40S:1)
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
source /pub/yournetid/setup_env.sh
conda activate llm_training

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Scratch dir: $TMPDIR"
echo "Results dir: $RESULTS_DIR"
echo "Start time: $(date)"

# Copy input files to scratch directory ($TMPDIR is automatically created)
echo "Copying input files to scratch..."
cp your_script.py data.csv $TMPDIR/
cp -r shared/src/models/ $TMPDIR/
cd $TMPDIR

# Function to sync results back to permanent storage
sync_results() {
    echo "Syncing results to permanent storage..."
    if [ -d "results" ]; then
        # Sync results back to your project area
        rsync -av --progress results/ /pub/kyvinhm/WobbleQuack/results/
        sync  # Force filesystem sync
        
        echo "✓ Results synced to /pub/kyvinhm/WobbleQuack/results/"
        
        # Quick summary
        echo "Result summary:"
        find results -name "*.csv" | head -5
        du -sh results
    else
        echo "⚠ Warning: No results directory found"
    fi
}

# Set up cleanup trap (runs on normal exit, cancellation, or failure)
trap sync_results EXIT

# Your actual work here
python your_script.py \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --data data.csv \
    --output_dir "$RESULTS_DIR/results/experiment_$SLURM_JOB_ID" \
    --batch_size 4 \
    --max_length 2048


echo "Job completed at: $(date)"
