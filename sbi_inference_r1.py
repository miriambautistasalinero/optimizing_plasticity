print("Inference round 1 P4")


import numpy as np
import torch
from Utils import *


from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform, RestrictionEstimator
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
#from sbi.inference.base import infer
from sbi import analysis as analysis
import pickle
#from sbi.inference import prepare_for_sbi
print("After all the imports")

# (S, B, D, A, mu, sigma, r_squared, R0, LS_e, LS_i)
# 0: S, 1: B, 2: D, 3: A, 4: mu, 5: sigma, 6: r_squared, 7: R0, 8: LS_e, 9: LS_i
## SBI REQUIREMENTS
# number of parameters
num_dim = 4 #8  

# Define priors
prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
# Ensure compliance with sbi's requirementsS
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# OPEN THE SIMULATIONS
# If it is saved as .npz
data_name = "P4_r1_fixed_exc_data" #nb_r1_sim_complete"
data = np.load(f'saved/sim/{data_name}.npz')

# If it is saved as pickle
# with open("saved/results/SBI_restrictedprior_plasticity_round1_test.pkl", "rb") as file:
    # restricted_prior = pickle.load(file)
print("Data loaded")    

# If the data is loaded but not simulated - Convert from numpy to torch
theta = torch.from_numpy(data['theta'])
x = torch.from_numpy(data['x'])
print(theta.shape)
print(x.shape)

## PROTOCOL 1

# Filter for r squared values > 0.9
x_rsqrt_filtered = x[x[:,6] >= 0.9]
theta_filtered = theta[x[:,6] >= 0.9]

# Take only rsqrt for it
x_filtered = x_rsqrt_filtered[:,6].unsqueeze(1)
print("x_filtered rsqrt PROTOCOL 1", x_filtered.shape)


## PROTOCOL 2
# Filter for population stability, balance and diversity
# Combine all filtering steps into one streamlined process
#notnan_mask = torch.isfinite(x).all(dim=1)
#balance_mask = x[:, 1] >= 0.9
#diversity_mask = x[:, 2] >= 0.9
#population_stability_mask = x[:, 12] >= 0.9

# Apply combined mask step-by-step to filter data
#combined_mask = notnan_mask  & balance_mask & diversity_mask
#combined_mask = notnan_mask & balance_mask & diversity_mask & population_stability_mask

# Filter x and theta arrays in one step
#x_filtered = x[combined_mask]
#theta_filtered = theta[combined_mask]
print("x_rsqrt_filtered", x_filtered.shape)
print("theta_rsqrt_filtered", theta_filtered.shape)
## INFERENCE

# 2. Train using S, and D

# Take S and B to train the neural density estimator
#x_filtered = x_rsqrt_filtere[:,6].unsqueeze(1)
# Taking S and D
#x_dbps = x_filtered[:,(1,2,12)]
#print("x_stab_balance", x_dbps.shape)

inference = SNPE(prior=prior, density_estimator="nsf")

# train the neural density estimator
density_estimator = inference.append_simulations(theta_filtered, x_filtered, proposal=prior).train()

print("Before posterior")
# Generate the posterior
posterior = inference.build_posterior(density_estimator)

# Save the posterior
# np.savez(f'saved/results/SBI_posterior_plasticity_round1.npz', posterior=posterior)
with open("saved/results/P4_r1_fixed_exc_posterior_rsqrt90.pkl", "wb") as f:
    pickle.dump(posterior, f)
print("First posterior saved")
