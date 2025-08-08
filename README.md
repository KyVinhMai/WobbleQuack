
# **🚀 QuickStart Guide: HPC3 for Language Modeling**

## **Overview**
A comprehensive language modeling pipeline for the Natural Language Processing group using UCI's HPC3 cluster. **Use It Or Lose It** - allocations are replenished every 6 months based on usage patterns (see [reallocation policy](https://hpc3.rcic.uci.edu)).

**📅 Pipeline Roadmap:**
- **Spring 2025**: Inference, Fine-tuning, & Interpretability Scripts
- **Summer 1st Half 2025**: Scripts dedicated to TransformerLens, LogitsLens, PicoLM, UnSloth etc.
- **Summer 2nd Half 2025**: Pretraining Scripts (125M-300M models) with Distributed Parallelization

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
# Highly Recommended: Set up SSH keys for easier access
```

### **2. Set Up Keys And Storage Space**
```
# 1. Go into setup/setup_cache_dir.sh and write your netid!
# 2. Now run the script, this will cache directories to /pub
bash setup/setup_cache_dir.sh

# 3. Now you should have your .env template and environment setup for every bash script 
```

### **3. Installation Setup (On Interactive Node!)**
```bash
# Get interactive node for setup, even if you're on VS Code
srun -p free --mem=32G --pty /bin/bash -i 

# Create conda environment with just Python
conda create --name llm_training python=3.10 numpy pandas tqdm pyyaml requests psutil -c conda-forge

# Activate environment
conda activate llm_training

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining packages with pip
pip install transformers>=4.40.0 datasets>=2.20.0 tokenizers>=0.20.0 safetensors>=0.4.0 huggingface_hub>=0.20.0 optimum>=1.20.0 pyarrow>=10.0.0 wandb>=0.16.0 minicons>=0.2.0 python-dotenv>=1.0.0
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
sbatch --partition=gpu --gres=gpu:A30:1 --A [labgpu] --time=00:05:00 --wrap="python test_gpu.py"
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