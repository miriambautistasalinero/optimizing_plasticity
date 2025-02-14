#!/bin/bash
#SBATCH -J P4_r3   # A single job name for the array

#SBATCH --nodes=1                  # Ensure that all cores are on one machine
##SBATCH --mem=400
##SBATCH -t 3-00:00                      # Maximum execution time (D-HH:MM)
#SBATCH -t 3-00:00:00                      # Maximum execution time (D-HH:MM)
##SBATCH --partition=h100-ferranti
#SBATCH --partition=a100-galvani                     # stands for --partition
##SBATCH --partition=cpu-galvani
##SBATCH --partition=2080ti-galvani
#SBATCH --gres=gpu:1             # optionally type and number of gpus
##SBATCH --partition=cpu-galvani
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=16             # Number of CPU cores per task

#SBATCH -o /mnt/qb/work/wu/wkn661/evolving_plasticity/logs/%A_%a.out  # Standard output
#SBATCH -e /mnt/qb/work/wu/wkn661/evolving_plasticity/logs/%A_%a.err  # Standard error
#SBATCH --array=0-12                # maps 1 to N to SLURM_ARRAY_TASK_ID below
#SBATCH --mail-type=END            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=miriam.bautista.neuro@gmail.com  # Email to which notifications will be sent


# Calculate the number of simulations per job
NUM_SIMULATIONS_PER_JOB=$((35000))

# Run the simulation script with the current array task ID and number of simulations per job
python sbi_simulate_r3.py $SLURM_ARRAY_TASK_ID $NUM_SIMULATIONS_PER_JOB
