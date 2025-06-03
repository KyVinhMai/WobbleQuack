# =============================================================================
# HuggingFace Cache Redirection Setup for HPC3
# =============================================================================

# 1. CREATE CACHE DIRECTORIES IN YOUR /pub SPACE
# Replace 'your_ucinetid' with your actual UCI NetID
mkdir -p /pub/your_ucinetid/huggingface_cache
mkdir -p /pub/your_ucinetid/huggingface_cache/transformers
mkdir -p /pub/your_ucinetid/huggingface_cache/datasets
mkdir -p /pub/your_ucinetid/huggingface_cache/hub

# 2. ADD ENVIRONMENT VARIABLES TO YOUR ~/.bashrc
echo "" >> ~/.bashrc
echo "# HuggingFace cache redirection to /pub space" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=/pub/your_ucinetid/huggingface_cache/transformers" >> ~/.bashrc
echo "export HF_HOME=/pub/your_ucinetid/huggingface_cache" >> ~/.bashrc
echo "export HF_HUB_CACHE=/pub/your_ucinetid/huggingface_cache/hub" >> ~/.bashrc
echo "export HUGGINGFACE_HUB_CACHE=/pub/your_ucinetid/huggingface_cache/hub" >> ~/.bashrc
echo "export HF_DATASETS_CACHE=/pub/your_ucinetid/huggingface_cache/datasets" >> ~/.bashrc

# 3. RELOAD YOUR BASH CONFIGURATION
source ~/.bashrc

# 4. VERIFY THE SETUP
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"

# =============================================================================
# JupyterLab and Related Cache Redirection Setup
# =============================================================================

# 1. CREATE ADDITIONAL CACHE DIRECTORIES
mkdir -p /pub/your_ucinetid/jupyter_cache
mkdir -p /pub/your_ucinetid/jupyter_cache/jupyter
mkdir -p /pub/your_ucinetid/jupyter_cache/ipython
mkdir -p /pub/your_ucinetid/jupyter_cache/matplotlib
mkdir -p /pub/your_ucinetid/jupyter_cache/torch

# 2. ADD MORE ENVIRONMENT VARIABLES TO ~/.bashrc
echo "" >> ~/.bashrc
echo "# JupyterLab and related cache redirection" >> ~/.bashrc
echo "export JUPYTER_CONFIG_DIR=/pub/your_ucinetid/jupyter_cache/jupyter" >> ~/.bashrc
echo "export JUPYTER_DATA_DIR=/pub/your_ucinetid/jupyter_cache/jupyter" >> ~/.bashrc
echo "export IPYTHONDIR=/pub/your_ucinetid/jupyter_cache/ipython" >> ~/.bashrc
echo "export MPLCONFIGDIR=/pub/your_ucinetid/jupyter_cache/matplotlib" >> ~/.bashrc
echo "export TORCH_HOME=/pub/your_ucinetid/jupyter_cache/torch" >> ~/.bashrc

# 3. SET UP BIOJHUB4 DIRECTORY REDIRECTION (for Jupyter Portal)
echo "export biojhub4HOME=/pub/your_ucinetid/biojhub4_dir" >> ~/.bashrc

# 4. RELOAD CONFIGURATION
source ~/.bashrc

# Check cache locations
echo $HF_HOME
ls -la /pub/$(whoami)/huggingface_cache/

# Check disk usage
du -sh /pub/$(whoami)/huggingface_cache/
df -h ~  # Should remain relatively empty