

print("Simulating round 1 for baseline network")
print("Inhibitory parameters are fixed")

import numpy as np
import torch
from Utils import *


from sbi.inference import NPE, SNPE, simulate_for_sbi
from sbi.utils import BoxUniform, RestrictionEstimator
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
#from sbi.inference.base import infer
from sbi import analysis as analysis
import pickle
import sys
#from sbi.inference import prepare_for_sbi
print("After all the imports")

# To manually parallelize
# Extract command line arguments
task_id = int(sys.argv[1])
num_simulations_per_job = int(sys.argv[2])
# Replace in num_simulations_per_job
# Replace in save f'saved/SBI_Results_lr_{task_id}.npz', theta=theta, x=x


## ROUND 1

# 1. Simulate plasticity rules
def simulator_neuron(parameters):
    p = parameters.cpu().detach().numpy()
    
    (S, B, D, A, mu, sigma, r_squared, R0, LD_e, LD_i, mean_B, mean_D, Population_Stability, cv_we, cv_wi, CV_sum,  pm_we, pm_wi) = run_baseline_network(p)
    return (S, B, D, A, mu, sigma, r_squared, R0, LD_e, LD_i, mean_B, mean_D, Population_Stability, cv_we, cv_wi, CV_sum,  pm_we, pm_wi)


# number of parameters
num_dim = 4 #8  

# Define priors
prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

print("Here")
# Ensure compliance with sbi's requirements
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(simulator_neuron, prior, prior_returns_numpy)

# Consistency check after making ready for sbi
check_sbi_inputs(simulator, prior)
#simulator, prior = prepare_for_sbi(simulator, prior)
print("Before simulating")

theta, x = simulate_for_sbi(simulator,
                            proposal=prior,
                            num_simulations=num_simulations_per_job,
                            num_workers=10,
                            show_progress_bar=True)

print("simulations done")
# Save simulations
#np.savez(f'saved/mini_sim/nb_r1_data_{task_id}.npz', theta=theta, x=x)
#file_id = task_id + 7
np.savez(f'saved/mini_sim/P4_r1_fixed_exc_data_{task_id}.npz', theta=theta, x=x)
print("First simulations saved")

# Check for NaN values
print(torch.sum(torch.isnan(theta)))
print(torch.sum(torch.isinf(theta)))

# Check for NaN and Inf values in 'x'
print(torch.sum(torch.isnan(x)))
print(torch.sum(torch.isinf(x)))

# Range of values of theta
u = theta.unique()
print("Parameter ranges from"  + str(u.min()) + " to " + str(u.max()))