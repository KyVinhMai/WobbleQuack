Here's a friendly, improved quickstart guide that emphasizes the key points while maintaining experiment integrity:

# **🚀 QuickStart Guide: HPC3 for Language Modeling**

## **Overview**
A comprehensive language modeling pipeline for the Natural Language Processing group using UCI's HPC3 cluster. **Use It Or Lose It** - allocations are replenished every 6 months based on usage patterns (see [reallocation policy](https://hpc3.rcic.uci.edu)).

**📅 Pipeline Roadmap:**
- **Spring 2025**: Inference, Fine-tuning, & Interpretability Scripts  
- **Summer 2025**: Pretraining Scripts (125M-300M models) with Distributed Parallelization

---

## **🎯 Essential Commands**
```bash
ssh [username]@hpc3.rcic.uci.edu          # Connect to cluster
sbank balance statement -u [username]      # Check GPU hour balance  
squeue -u [username]                       # Monitor your jobs
scancel [job_id]                           # Cancel a job if needed
dfsquotas [username] all                   # Check storage quota for users
df -h ~                                    # How much storage is used/available in /data/home 
```

---

## **🚨 Critical Rules (Avoid Account Suspension!)**

### **❌ NEVER on Login Nodes:**
- Computational jobs or GPU work
- Jobs running >1 hour or using significant CPU/memory
- Multi-threaded compilation (`make -j 8`)
- Conda/pip installations 
- Large file downloads (>few GB)

### **❌ NEVER with Modules:**
- Load modules in `.bashrc` or `.bash_profile` 
- Unload auto-loaded prerequisite modules
- **Always specify version**: `module load python/3.10.2` ✅ not `module load python` ❌

---

## **⚡ Quick Setup**

### **1. First-Time Login**
```bash
ssh [your-ucinetid]@hpc3.rcic.uci.edu
# Use UCI password + DUO authentication
# Pro tip: Set up SSH keys for easier access
```

### **2. Set Up Keys And Storage Space**
```
# 1. Create .env file in your project directory
# /pub/your_ucinetid/your_project/.env

HF_TOKEN=hf_your_actual_token_here
WANDB_API_KEY=your_wandb_key_here
OPENAI_API_KEY=your_openai_key_here

# 2. Load .env in your Python code
# install python-dotenv: pip install python-dotenv

# 3. Alternative: Set environment variables in your Slurm script
# This is often preferred on HPC systems
```

### **3. Environment Setup (On Interactive Node!)**
```bash
# Get interactive node for setup
srun -p free --mem=32G --pty /bin/bash -i

# Load essential modules
module load python/3.10.2
module load cuda/11.7.1

# Initialize conda (one-time only)
conda init bash
source ~/.bashrc

# Create your environment
conda create -n llm_experiments python=3.10
conda activate llm_experiments

# Install core packages (NO quantization for experiment integrity!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers accelerate datasets tqdm pandas numpy
pip install huggingface_hub wandb  # For model access & experiment tracking

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### **4. Quick Test Job**
```bash
# Create simple test script
cat > test_gpu.py << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

# Submit test job
sbatch --partition=free-gpu --gres=gpu:V100:1 --time=00:05:00 --wrap="python test_gpu.py"
```

---

## **💡 Pro Tips**
- **Check quotas regularly**: `dfsquotas [username] all`
- **Use /pub/[username] for data**: Never store large files in $HOME
- **Monitor jobs**: Use `seff [job_id]` to check efficiency after completion
- **Request appropriate resources**: Don't over-request memory/GPUs you won't use

---

## **📋 Next Steps**
1. **Read the detailed guides** (coming soon): Job submission (https://www.notion.so/Quick-Guide-SLURM-HPC3-204a2623684480f1bd06fb40931df587?source=copy_link), storage management, debugging
2. **Ask for help**: Check with lab for HPC3 support sessions
3. **Bookmark**: [HPC3 Documentation](https://hpc3.rcic.uci.edu) & [Roy's CS277 Guide](https://royf.org/crs/CS277/W24/HPC3.pdf)

---
**🎓 Remember**: HPC3 is a shared resource. Be considerate of others, follow the rules, and happy computing! 

**Questions?** Ask in lab Slack or submit tickets to `hpc-support@uci.edu`