## 🎯 Overview

High-level description of the technical implementation, architecture decisions, and key components.

## **QuickStart Guide To HPC3 for Language Modeling:**

You should be attached to a lab and given a set of GPU allocated hours

`ssh [username]@hpc3.rcic.edu` - Log onto server

`sbank balance statement -u [username]` -  Check your available balance

`squeue -u [username]` - Check your submitted jobs

🚨Absolute No’s🚨: 

<aside>
💡

Never load modules in your .bashrc or .bash_profilefiles.

Never unload modules that were auto-loaded by a module itself

Do not run on login node:

- **any computational jobs**
- **any job that runs for more than 1hr or is using significant memory and CPU**
- **any compilation** that asks for multiple threads while running make
(for example `make -j 8`)
- **any conda or R installation** of packages or environments
- **any downloads** of packages, data, large files that exceed a few Gb.
</aside>

1.  How to log on:
    1. SSH [your UCI net ID]@`hpc3.rcic.uci.edu`
    Your password is your UCInetID password.
    2. Respond to multi-factor authentication prompts
    3. Use ssh key (highly recommended)

Set Up

1. Set up and add Conda to bash files (https://rcic.uci.edu/software/user-installed.html#install-with-conda)
    1. NOTE: You should not try to install packages directly on the login node! Use an interactive node 
    
    ```bash
    srun -p free --mem=32G --pty /bin/bash -i
    
    # Load modules
    module load python/3.10.2
    module load cuda/11.7.1
    
    # Create conda enviornment
    conda init bash
    conda create -n [You Environment Name]
    
    # Activate environment
    conda activate [You Environment Name]
    
    # Install PyTorch with matching CUDA version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    
    pip install transformers accelerate bitsandbytes tqdm
    
    #Check to see if it is installed
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
    ```
    

## 📋 Additional Resources

https://royf.org/crs/CS277/W24/HPC3.pdf
