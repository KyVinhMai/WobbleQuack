#!/bin/bash
#SBATCH --job-name=attention_analysis
#SBATCH --partition=gpu               
#SBATCH --account=RFUTRELL_LAB_GPU     
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00                
#SBATCH --nodes=1
#SBATCH --output=logs/attention_analysis_%j.out
#SBATCH --error=logs/attention_analysis_%j.err

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "Setting up environment..."
module load python/3.10.2
module load cuda/11.7.1


source /pub/kyvinhm/setup_env.sh
eval "$(conda shell.bash hook)"
conda activate

# --- Job Information ---
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Running on node: $SLURMD_NODENAME"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Working directory (submission): $SLURM_SUBMIT_DIR"
echo "Temporary directory (scratch): $TMPDIR"
echo "Start time: $(date)"

echo "Preparing scratch space..."
JOB_SCRATCH_OUTPUT_DIR="$TMPDIR/attention_analysis_outputs"
mkdir -p "$JOB_SCRATCH_OUTPUT_DIR"

echo "Copying project files to scratch..."
cp -r "$SLURM_SUBMIT_DIR/interpretability" "$TMPDIR/"
cp -r "$SLURM_SUBMIT_DIR/shared" "$TMPDIR/"    


cd $TMPDIR


PERMANENT_RESULTS_DIR="/pub/kyvinhm/WobbleQuack/results/interpretability/job_${SLURM_JOB_ID}"
mkdir -p "$PERMANENT_RESULTS_DIR"
echo "Permanent results will be stored in: $PERMANENT_RESULTS_DIR"

sync_results() {
    echo "--- Syncing results to permanent storage ---"
    if [ -d "$JOB_SCRATCH_OUTPUT_DIR" ] && [ "$(ls -A $JOB_SCRATCH_OUTPUT_DIR)" ]; then
        echo "Found results in $JOB_SCRATCH_OUTPUT_DIR. Copying..."
        rsync -av --progress "$JOB_SCRATCH_OUTPUT_DIR/" "$PERMANENT_RESULTS_DIR/"
        sync
        echo "Results successfully synced to $PERMANENT_RESULTS_DIR"
        echo "Contents of synced results:"
        ls -R "$PERMANENT_RESULTS_DIR"
    else
        echo "Warning: No results found in $JOB_SCRATCH_OUTPUT_DIR to sync, or directory is empty."
    fi
    echo "--- Sync complete at $(date) ---"
}

trap sync_results EXIT

echo "Starting Python attention analysis script..."
python interpretability/analyze_attention.py \
    --model_identifier "openai-community/gpt2" \
    --input_text "Hello world, this is a test of the attention mechanism." \
    --layers "0" "5" "11" \
    --heads "0" "1" \
    --output_dir "$JOB_SCRATCH_OUTPUT_DIR" \
    --device "cuda" \
    --hf_token "${HF_TOKEN}"

echo "Job completed at: $(date)"
