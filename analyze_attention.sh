#!/bin/bash
#SBATCH --job-name=attention_analysis
#SBATCH --partition=gpu
#SBATCH --account=RFUTRELL_LAB_GPU
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --output=logs/attention_analysis_%j.out
#SBATCH --error=logs/attention_analysis_%j.err

mkdir -p logs
module load python/3.10.2
module load cuda/11.7.1

source /pub/kyvinhm/setup_env.sh
conda activate llm_training

cp interpretability/analyze_attention.py $TMPDIR/
cp -r shared/src/models/ $TMPDIR/
cd $TMPDIR

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

trap sync_results EXIT

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

python analyze_attention.py \
    --model_identifier "gpt2" \
    --input_text "Hello world, this is a test of the attention mechanism." \
    --layers "0" "5" "11" \
    --heads "0" "1" \
    --output_dir "llm-pipeline/interpretability/outputs/gpt2_hello_attention" \
    --device "cuda"

echo "Job completed at: $(date)"
