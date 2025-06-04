#!/bin/bash
# =============================================================================
# HPC3 Cache and Environment Setup Script
# =============================================================================
# This script sets up cache directories and environment variables to avoid
# filling up your $HOME directory and resolve common warnings on HPC3
# 
# Usage: 
#   1. Edit the USER_ID variable below to match your UCI NetID
#   2. Run: bash setup_hpc3_cache.sh
# =============================================================================

# CONFIGURATION - EDIT THIS
USER_ID="yournetid"  # Replace with your UCI NetID
PUB_DIR="/pub/${USER_ID}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}HPC3 Cache and Environment Setup for ${USER_ID}${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# =============================================================================
# 1. VERIFY PUB DIRECTORY EXISTS
# =============================================================================
if [ ! -d "$PUB_DIR" ]; then
    echo -e "${RED}ERROR: $PUB_DIR does not exist!${NC}"
    echo -e "${RED}Please check your UCI NetID and ensure you have access to /pub space${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Verified access to $PUB_DIR${NC}"

# =============================================================================
# 2. CREATE CACHE DIRECTORIES
# =============================================================================
echo -e "${YELLOW}Creating cache directories...${NC}"

# HuggingFace cache directories
mkdir -p ${PUB_DIR}/cache/huggingface/{transformers,datasets,hub,models}
mkdir -p ${PUB_DIR}/cache/torch/hub
mkdir -p ${PUB_DIR}/cache/torch/checkpoints

# Python package caches
mkdir -p ${PUB_DIR}/cache/pip
mkdir -p ${PUB_DIR}/cache/conda/pkgs
mkdir -p ${PUB_DIR}/cache/conda/envs

# Jupyter and IPython
mkdir -p ${PUB_DIR}/cache/jupyter/{jupyter,ipython,matplotlib,plotly}

# Development tools
mkdir -p ${PUB_DIR}/cache/tensorboard
mkdir -p ${PUB_DIR}/cache/wandb

# Create local bin directory for user installations
mkdir -p ${PUB_DIR}/local/bin

# Biojhub4 directory (for HPC3 Jupyter Portal)
mkdir -p ${PUB_DIR}/biojhub4_dir

echo -e "${GREEN}✓ Created all cache directories${NC}"

# =============================================================================
# 3. CREATE BACKUP OF EXISTING .bashrc
# =============================================================================
if [ -f ~/.bashrc ]; then
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    echo -e "${GREEN}✓ Backed up existing .bashrc${NC}"
fi

# =============================================================================
# 4. CREATE ENVIRONMENT SETUP SCRIPT (RECOMMENDED APPROACH)
# =============================================================================
ENV_SCRIPT="${PUB_DIR}/setup_env.sh"

cat > "$ENV_SCRIPT" << EOF
#!/bin/bash
# =============================================================================
# HPC3 Environment Setup for ${USER_ID}
# Contains all your project-specific environment variables
# Source this script instead of modifying ~/.bashrc directly
# Usage: source ${PUB_DIR}/setup_env.sh
# =============================================================================

# HuggingFace Cache Configuration
export HF_HOME="${PUB_DIR}/cache/huggingface"
export TRANSFORMERS_CACHE="${PUB_DIR}/cache/huggingface/transformers"
export HF_HUB_CACHE="${PUB_DIR}/cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="${PUB_DIR}/cache/huggingface/hub"
export HF_DATASETS_CACHE="${PUB_DIR}/cache/huggingface/datasets"

# PyTorch Cache Configuration
export TORCH_HOME="${PUB_DIR}/cache/torch"
export TORCH_HUB="${PUB_DIR}/cache/torch/hub"

# Python Package Cache Configuration
export PIP_CACHE_DIR="${PUB_DIR}/cache/pip"
export CONDA_PKGS_DIRS="${PUB_DIR}/cache/conda/pkgs"

# Jupyter Configuration
export JUPYTER_CONFIG_DIR="${PUB_DIR}/cache/jupyter/jupyter"
export JUPYTER_DATA_DIR="${PUB_DIR}/cache/jupyter/jupyter"
export IPYTHONDIR="${PUB_DIR}/cache/jupyter/ipython"
export MPLCONFIGDIR="${PUB_DIR}/cache/jupyter/matplotlib"
export PLOTLY_CACHE_DIR="${PUB_DIR}/cache/jupyter/plotly"

# Development Tools
export TENSORBOARD_LOG_DIR="${PUB_DIR}/cache/tensorboard"
export WANDB_CACHE_DIR="${PUB_DIR}/cache/wandb"
export WANDB_DATA_DIR="${PUB_DIR}/cache/wandb"

# Biojhub4 Directory (for HPC3 Jupyter Portal)
export biojhub4HOME="${PUB_DIR}/biojhub4_dir"

# Add local bin to PATH (fixes pip install warnings)
export PATH="${PUB_DIR}/local/bin:\$HOME/.local/bin:\$PATH"

# Python Path Configuration
export PYTHONPATH="${PUB_DIR}:\$PYTHONPATH"

# Set Python to not create __pycache__ directories (optional)
export PYTHONDONTWRITEBYTECODE=1

# CUDA Configuration (if using GPUs)
export CUDA_CACHE_PATH="${PUB_DIR}/cache/cuda"
mkdir -p "\$CUDA_CACHE_PATH"

echo "✓ Environment configured for ${USER_ID}"
echo "✓ Cache directories set to ${PUB_DIR}/cache/"
echo "✓ Local bin directory: ${PUB_DIR}/local/bin"

EOF

chmod +x "$ENV_SCRIPT"
echo -e "${GREEN}✓ Created environment setup script: $ENV_SCRIPT${NC}"

# =============================================================================
# 5. CREATE PROJECT TEMPLATE SLURM SCRIPT
# =============================================================================
TEMPLATE_SLURM="${PUB_DIR}/template_job.slurm"

cat > "$TEMPLATE_SLURM" << 'EOF'
#!/bin/bash
#SBATCH --job-name=my_experiment
#SBATCH --partition=gpu
#SBATCH --account=YOUR_LAB_GPU_ACCOUNT
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Load required modules (do this in job scripts, not ~/.bashrc)
module load python/3.10.2
module load cuda/11.7.1

# Source your environment setup
source /pub/YOUR_UCI_NETID/setup_env.sh

# Activate conda environment
conda activate your_env_name

# Set up logging directory
mkdir -p logs

# Run your experiment
python your_script.py
EOF

echo -e "${GREEN}✓ Created Slurm job template: $TEMPLATE_SLURM${NC}"

# =============================================================================
# 6. CREATE .ENV TEMPLATE FOR PYTHON PROJECTS
# =============================================================================
ENV_TEMPLATE="${PUB_DIR}/.env.template"

cat > "$ENV_TEMPLATE" << EOF
# Environment variables for Python projects
# Copy this to your project directory as .env

# HuggingFace Token
HF_TOKEN=hf_your_actual_token_here
WANDB_API_KEY=your_wandb_key_here
OPENAI_API_KEY=your_openai_key_here

# Project Configuration
PROJECT_ROOT=${PUB_DIR}/your_project
RESULTS_DIR=${PUB_DIR}/your_project/results
DATA_DIR=${PUB_DIR}/your_project/data

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Python Configuration
PYTHONPATH=${PUB_DIR}/your_project
PYTHONDONTWRITEBYTECODE=1
EOF

echo -e "${GREEN}✓ Created .env template: $ENV_TEMPLATE${NC}"

# =============================================================================
# 7. OPTIONAL: ADD SOURCE LINE TO .bashrc (MINIMAL APPROACH)
# =============================================================================
echo ""
echo -e "${YELLOW}Choose your setup approach:${NC}"
echo -e "${BLUE}1. Manual (recommended): Source the environment script when needed${NC}"
echo -e "${BLUE}2. Automatic: Add source line to ~/.bashrc (against HPC3 best practices)${NC}"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "2" ]; then
    echo "" >> ~/.bashrc
    echo "# Source HPC3 environment setup (added by setup script)" >> ~/.bashrc
    echo "source ${ENV_SCRIPT}" >> ~/.bashrc
    echo -e "${YELLOW}⚠ Added source line to ~/.bashrc${NC}"
    echo -e "${YELLOW}⚠ Note: HPC3 best practices recommend loading modules in job scripts instead${NC}"
else
    echo -e "${GREEN}✓ Manual approach selected (recommended)${NC}"
fi

# =============================================================================
# 8. CREATE USAGE INSTRUCTIONS
# =============================================================================
INSTRUCTIONS="${PUB_DIR}/SETUP_INSTRUCTIONS.md"

cat > "$INSTRUCTIONS" << EOF
# HPC3 Environment Setup Instructions

## Quick Start

### For Interactive Work:
\`\`\`bash
# Log into HPC3
ssh your_uci_netid@hpc3.rcic.uci.edu

# Load modules
module load python/3.10.2
module load cuda/11.7.1

# Source environment
source ${PUB_DIR}/setup_env.sh

# Activate conda environment
conda activate your_env_name
\`\`\`

### For Batch Jobs:
1. Copy and edit the template: \`cp ${TEMPLATE_SLURM} my_job.slurm\`
2. Update the account, environment name, and script path
3. Submit: \`sbatch my_job.slurm\`

### For Python Projects:
1. Copy the .env template: \`cp ${ENV_TEMPLATE} your_project/.env\`
2. Edit the .env file with your actual values
3. Use \`python-dotenv\` to load environment variables

## Cache Locations
- HuggingFace: \`${PUB_DIR}/cache/huggingface/\`
- PyTorch: \`${PUB_DIR}/cache/torch/\`
- Pip: \`${PUB_DIR}/cache/pip/\`
- Jupyter: \`${PUB_DIR}/cache/jupyter/\`

## Monitoring Usage
\`\`\`bash
# Check cache sizes
du -sh ${PUB_DIR}/cache/*

# Check home directory usage (should stay small)
df -h ~

# Check pub directory quota
dfsquotas \$(whoami) dfs6
\`\`\`

## Troubleshooting
- If you get "command not found" errors, check your PATH
- If packages install to wrong location, verify cache environment variables
- If jobs fail, check that modules are loaded in job script, not ~/.bashrc
EOF

echo -e "${GREEN}✓ Created setup instructions: $INSTRUCTIONS${NC}"

# =============================================================================
# 9. VERIFICATION AND SUMMARY
# =============================================================================
echo ""
echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}SETUP COMPLETE!${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Test the environment setup
source "$ENV_SCRIPT"

echo -e "${GREEN}Environment Variables Set:${NC}"
echo "  HF_HOME: $HF_HOME"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  PIP_CACHE_DIR: $PIP_CACHE_DIR"
echo "  PATH additions: ${PUB_DIR}/local/bin"

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Source the environment: ${BLUE}source $ENV_SCRIPT${NC}"
echo "2. Install python-dotenv: ${BLUE}pip install python-dotenv${NC}"
echo "3. Read instructions: ${BLUE}cat $INSTRUCTIONS${NC}"
echo "4. Test with a small script to verify cache redirection works"

echo ""
echo -e "${YELLOW}Files Created:${NC}"
echo "  - Environment script: $ENV_SCRIPT"
echo "  - Slurm template: $TEMPLATE_SLURM"
echo "  - .env template: $ENV_TEMPLATE"
echo "  - Instructions: $INSTRUCTIONS"

echo ""
echo -e "${GREEN}✓ All cache directories created and configured!${NC}"
echo -e "${GREEN}✓ Your \$HOME directory should now stay clean${NC}"