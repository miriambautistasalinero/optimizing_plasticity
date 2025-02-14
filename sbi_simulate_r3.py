# sbi simulations round 2 with loaded previous posterior
print("Simulations round 3 PROTOCOL 4")

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
#from sbi.inference import prepare_for_sbis
print("After all the imports")

# To manually parallelize
# Extract command line arguments
task_id = int(sys.argv[1])
num_simulations_per_job = int(sys.argv[2])
# Replace in num_simulations_per_job
# Replace in save f'saved/SBI_Results_lr_{task_id}.npz', theta=theta, x=x


## SBI BASICS
print("Before SBI Basics")
# 1. Simulate plasticity rules
def simulator_neuron(parameters):
    p = parameters.cpu().detach().numpy()
    (S, B, D, A, mu, sigma, r_squared, R0, LD_e, LD_i, mean_B, mean_D, Population_Stability, cv_we, cv_wi, CV_sum,  pm_we, pm_wi) = run_baseline_network(p)
    return (S, B, D, A, mu, sigma, r_squared, R0, LD_e, LD_i, mean_B, mean_D, Population_Stability, cv_we, cv_wi, CV_sum,  pm_we, pm_wi)

# number of parameters
num_dim = 4 #8  

# Define priors
prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

## LOAD POSTERIOR


print("Here")
# Ensure compliance with sbi's requirements
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(simulator_neuron, prior, prior_returns_numpy)

# Consistency check after making ready for sbi
check_sbi_inputs(simulator, prior)
print("End SBI Basics")
## -- SBI BASICS

## LOAD POSTERIOR

# Load pickle data
posterior_name = "P4_r2_fixed_exc_posterior_rsqrt90_stabil90"

with open(f"saved/results/{posterior_name}.pkl", "rb") as file:
    posterior_r2 = pickle.load(file)

print("Posterior round 2 loaded", posterior_r2)

# observation of S = 1 and B = 1, D = 1
#observation = torch.zeros(1, ) + 1 

## PROTOCOL 1
# pop stability = 1, balance = 1
observation = torch.zeros(1, ) + 1

# Proposal - the samples from the posterior become the new prior (pi_2)
prior_3 =  posterior_r2.set_default_x(observation)



## SIMULATE DATA
theta_3, x_3 = simulate_for_sbi(simulator,
                            proposal= prior_3,
                            num_simulations=num_simulations_per_job,
                            num_workers=10,
                            show_progress_bar=True)

print("simulations done")
# Save simulations

np.savez(f'saved/mini_sim/P4_r3_fixed_exc_data_rsqrt90_stabil90_{task_id}.npz', theta=theta_3, x=x_3)
print("Second simulations saved")
