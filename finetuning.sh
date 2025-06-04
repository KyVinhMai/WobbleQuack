#!/bin/bash
#SBATCH --job-name=Finetuning_GPT2
#SBATCH --partition=gpu
#SBATCH --account=RFUTRELL_LAB_GPU
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/Finetuning_GPT2_%j.out
#SBATCH --error=logs/Finetuning_GPT2_%j.err

mkdir -p logs

module load python/3.10.2
module load cuda/11.7.1

source ~/.bashrc
source /pub/kyvinhm/setup_env.sh
conda activate llm_training

cp interpretability/analyze_attention.py $TMPDIR/
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

python llm-pipeline/training/scripts/python/run_finetuning.py \
    --model_name_or_path "gpt2" \
    --dataset_path "./path/to/your/train.txt" \
    --text_column "text" \
    --output_dir "$RESULTS_DIR/gpt2_lora_finetuned" \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --max_seq_length 512 \
    --logging_steps 10 \
    --save_steps 200 \
    --fp16 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "c_attn" \
    --report_to "wandb" 

echo "Job completed at: $(date)"