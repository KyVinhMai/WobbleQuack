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

source ~/.bashrc
conda activate llm_experiments

export CUDA_VISIBLE_DEVICES=0

export TMPDIR_CUSTOM=$TMPDIR/experiment_$SLURM_JOB_ID
mkdir -p $TMPDIR_CUSTOM

cd $TMPDIR_CUSTOM

# Print job info for debugging
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

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
