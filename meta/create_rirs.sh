#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=16000
#SBATCH --time=24:00:00
#SBATCH --mail-user=jwravenscroft1@sheffield.ac.uk

#load the modules
module load Anaconda3/5.3.0
module load fosscuda/2019b  # includes GCC 8.3
module load imkl/2019.5.281-iimpi-2019b
module load CMake/3.15.3-GCCcore-8.3.0
#python environment
source activate speechbrain

srun --export=ALL python3 create_whamr_rirs.py --output-dir ~/fastdata/data/whamr/rirs
