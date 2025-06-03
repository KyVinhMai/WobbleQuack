#!/bin/bash
#SBATCH --job-name=Finetuning_GPT2
#SBATCH --partition=gpu
#SBATCH --account=RFUTRELL_LAB_GPU
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=Finetuning_GPT2_%j.out
#SBATCH --error=Finetuning_GPT2_%j.err

module load python/3.10.2
module load cuda/11.7.1

source ~/.bashrc
conda activate ambi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python llm-pipeline/training/scripts/python/run_finetuning.py \
    --model_name_or_path "gpt2" \
    --dataset_path "./path/to/your/train.txt" \
    --text_column "text" \
    --output_dir "./gpt2_lora_finetuned" \
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

# Copy results back (only if the output file exists)
if [ -f output.txt ]; then
    cp output.txt $SLURM_SUBMIT_DIR/output_${SLURM_JOB_ID}.txt
else
    echo "Warning: output.txt was not created"
fi

rm -rf $TMPDIR/huggingface