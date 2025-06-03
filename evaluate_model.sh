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

source ~/.bashrc
conda activate ambi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

cp input.txt llama7b_simple_inference.py $TMPDIR/
cd $TMPDIR

python evaluate_model_response.py \
    --input_file \
    --models \
    --max_generatation \
    --extract_logprobs \
    --max_tasks \
    --fast_inference \
    --output_dir \

# Copy results back (only if the output file exists)
if [ -f output.txt ]; then
    cp output.txt $SLURM_SUBMIT_DIR/output_${SLURM_JOB_ID}.txt
else
    echo "Warning: output.txt was not created"
fi

rm -rf $TMPDIR/huggingface