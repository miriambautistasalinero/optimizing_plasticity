print("Hello")

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
from tqdm import tqdm
import sys
#from sbi.inference import prepare_for_sbi
print("After all the imports")

# To manually parallelize
# Extract command line arguments
task_id = int(sys.argv[1])
num_simulations_per_job = int(sys.argv[2])


## Basic SBI commands
# 1. Simulate plasticity rules
def simulator_neuron(parameters):
    p = parameters.cpu().detach().numpy()
    (S, B, D, A, mu, sigma, r_squared) = run_network_plasticity(p)
    return (S, B, D, A, mu, sigma, r_squared)


# number of parameters
num_dim = 8  

# Define priors
prior = BoxUniform(low=-1 * torch.ones(num_dim), high=1 * torch.ones(num_dim))

print("Here")
# Ensure compliance with sbi's requirements
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(simulator_neuron, prior, prior_returns_numpy)

# Consistency check after making ready for sbi
check_sbi_inputs(simulator, prior)
#simulator, prior = prepare_for_sbi(simulator, prior)
print("Basics check")

## -- END BASICS

# LOAD 
with open("saved/results/SBI_restrictedprior_plasticity_r1_1HT.pkl", "rb") as file:
    restricted_prior = pickle.load(file)


# Generate the new theta and x from the restricted prior
new_theta, new_x = simulate_for_sbi(simulator, restricted_prior, num_simulations=num_simulations_per_job,
                            num_workers=10,
                            show_progress_bar=True)
print("New simulations done")

with open("saved/mini_sim/SBI_datarestricted_plasticity_r1_1HT.pkl", "wb") as f:
    pickle.dump({'theta': theta, 'x': x}, f)
