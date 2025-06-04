#!/bin/bash
#SBATCH --job-name=Evaluate_Model
#SBATCH --partition=gpu
#SBATCH --account=RFUTRELL_LAB_GPU
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --tmp=10GB
#SBATCH --output=logs/Evaluate_Model_%j.out
#SBATCH --error=logs/Evaluate_Model_%j.err

mkdir -p logs

module load python/3.10.2
module load cuda/11.7.1

eval "$(conda shell.bash hook)"
source /pub/kyvinhm/setup_env.sh
conda activate carc

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

echo "Preparing scratch space..."
JOB_SCRATCH_OUTPUT_DIR="$TMPDIR/model_evaluation_outputs"
mkdir -p "$JOB_SCRATCH_OUTPUT_DIR"

echo "Copying project files to scratch..."
cp -r "$SLURM_SUBMIT_DIR/evaluation" "$TMPDIR/" # Copies the 'evaluation' folder
cp -r "$SLURM_SUBMIT_DIR/shared" "$TMPDIR/"     # Copies the 'shared' folder for ModelLoader

cd $TMPDIR 


PERMANENT_RESULTS_DIR="/pub/kyvinhm/WobbleQuack/results/evaluation/job_${SLURM_JOB_ID}"
mkdir -p "$PERMANENT_RESULTS_DIR"
echo "Permanent results will be stored in: $PERMANENT_RESULTS_DIR"

# --- Sync Results Function ---
# This function will be called on exit to copy results from scratch to permanent storage
sync_results() {
    echo "--- Syncing results to permanent storage ---"
    if [ -d "$JOB_SCRATCH_OUTPUT_DIR" ] && [ "$(ls -A $JOB_SCRATCH_OUTPUT_DIR)" ]; then
        echo "Found results in $JOB_SCRATCH_OUTPUT_DIR. Copying..."
        rsync -av --progress "$JOB_SCRATCH_OUTPUT_DIR/" "$PERMANENT_RESULTS_DIR/"
        # Ensure data is written to disk
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

echo "Starting Python evaluation script..."
python evaluation/evaluate_model_response.py \
    --stimuli_csv "evaluation/syntactic_stimuli.csv" \
    --sentence_column "sentence" \
    --model_identifiers "openai-community/gpt2" "EleutherAI/pythia-70m" \
    --output_dir "$JOB_SCRATCH_OUTPUT_DIR" \
    --mode "sentence" \
    --hf_token "${HF_TOKEN}" \
    --device "cuda" \
    --scorer_device "cuda" \
    --fast_inference \
    # --quantization_bits 4 \ # Optional: Uncomment and set to 4 or 8 if needed
    # --trust_remote_code \   # Optional: Uncomment if models require trusting remote code
                              # IMPORTANT: If using --trust_remote_code, ensure
                              # evaluate_model_response.py is updated to accept this argument.

echo "Python script finished."
echo "Job completed at: $(date)"
