#!/bin/bash
#SBATCH -J pipeline    # A single job name for the array

#SBATCH --nodes=1                  # Ensure that all cores are on one machine
##SBATCH --mem=400
##SBATCH -t 3-00:00                      # Maximum execution time (D-HH:MM)
#SBATCH -t 2-00:00:00                      # Maximum execution time (D-HH:MM)
#SBATCH --partition=a100-galvani                     # stands for --partition
##SBATCH --partition=cpu-galvani
##SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1             # optionally type and number of gpus
##SBATCH --partition=cpu-galvani
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=16             # Number of CPU cores per task

#SBATCH -o /mnt/qb/work/wu/wkn661/evolving_plasticity/logs/%A_%a.out  # Standard output
#SBATCH -e /mnt/qb/work/wu/wkn661/evolving_plasticity/logs/%A_%a.err  # Standard error
##SBATCH --array=0-40                  # maps 1 to N to SLURM_ARRAY_TASK_ID below
#SBATCH --mail-type=END            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=miriam.bautista.neuro@gmail.com  # Email to which notifications will be sent



python sbi_completepipeline.py


