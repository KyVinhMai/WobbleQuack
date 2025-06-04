#!/bin/bash
#SBATCH --job-name=Evaluate_Model
#SBATCH --partition=free-gpu
#SBATCH --account=RFUTRELL_LAB_GPU
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --tmp=10GB
#SBATCH --output=Evaluate_Model_%j.out
#SBATCH --error=Evaluate_Model_%j.err

module load python/3.10.2
module load cuda/11.7.1

source /pub/kyvinhm/setup_env.sh
conda activate llm_training

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

cp -r evaluation $TMPDIR/
cd $TMPDIR

python evaluate_model_response.py \
    --stimuli_csv syntatic_stimmuli.csv \
    --sentence_column sentence \
    --model_identifiers \
    --hf_token YOUR_HF_TOKEN_IF_NEEDED \
    --output_dir llm-pipeline/evaluation/results/logprob_tests \
    --mode sentence

# Copy results back (only if the output file exists)
if [ -f output.txt ]; then
    cp output.txt $SLURM_SUBMIT_DIR/output_${SLURM_JOB_ID}.txt
else
    echo "Warning: output.txt was not created"
fi

rm -rf $TMPDIR/huggingface