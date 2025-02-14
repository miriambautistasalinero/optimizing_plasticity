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
#from sbi.inference import prepare_for_sbi
print("After all the imports")


## ROUND 1

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
print("Before simulating")

# Do simulations
theta, x = simulate_for_sbi(simulator,
                            proposal=prior,
                            num_simulations=1000,
                            num_workers=10,
                            show_progress_bar=True)

print("simulations done")

# Save simulations
#np.savez(f'saved/sim/SBI_data_plasticity_round1.npz', theta=theta, x=x)
with open("saved/sim/SBI_data_pipeline_round1_test.pkl", "wb") as f:
    pickle.dump({'theta': theta, 'x': x}, f)
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

# 2. Train using S and B 

# Take S and B to train the neural density estimator
x_stab_balance = x[:,0:2]

inference = NPE(prior=prior)

# train the neural density estimator
density_estimator = inference.append_simulations(theta, x_stab_balance, proposal=prior).train()

# Generate the posterior
posterior = inference.build_posterior(density_estimator)

# Save the posterior
# np.savez(f'saved/results/SBI_posterior_plasticity_round1.npz', posterior=posterior)
with open("saved/results/SBI_posterior_pipeline_round1_test.pkl", "wb") as f:
    pickle.dump(posterior, f)
print("First posterior saved")

# Sample from posterior as prior for observation (S, B) = (0,0)

# observation of S = 0 and B = 0
observation = torch.zeros(2, )

# Proposal - the samples from the posterior become the new prior (pi_2)
prior_2 =  posterior.set_default_x(observation)

## ROUND 2

# 1. Simulate from the new prior
print("Before second simulations")
theta_2, x_2 = simulate_for_sbi(simulator,
                            proposal=prior_2,
                            num_simulations=1000,
                            num_workers=10,
                            show_progress_bar=True)

# Save simulations round 2
# np.savez(f'saved/sim/SBI_data_plasticity_round2.npz', theta=theta, x=x)
with open("saved/sim/SBI_data_pipeline_round2_test.pkl", "wb") as f:
    pickle.dump({'theta': theta_2, 'x': x_2}, f)
print("Second round simulations saved")

# Check for NaN values
print(torch.sum(torch.isnan(theta_2)))
print(torch.sum(torch.isinf(theta_2)))

# Check for NaN and Inf values in 'x'
print(torch.sum(torch.isnan(x_2)))
print(torch.sum(torch.isinf(x_2)))

# Range of values of theta
u = theta.unique()
print("Parameter ranges from"  + str(u.min()) + " to " + str(u.max()))

# 2. Take sigma to train the neural density estimator
x_sigma = x_2[:,5].unsqueeze(1)

# 3. Train using sigma
inference_2 = SNPE(density_estimator="nsf")

# train the neural density estimator
density_estimator_2 = inference_2.append_simulations(theta_2, x_sigma).train()

# Generate the posterior
posterior_final = inference_2.build_posterior(density_estimator_2)

# Save the posterior
#np.savez(f'saved/results/SBI_posterior_plasticity_final.npz', posterior=posterior_final)
with open("saved/results/SBI_posterior_pipeline_round2_test.pkl", "wb") as f:
    pickle.dump(posterior_final, f)
print("Second posterior saved")


# From this, i will have to load the posterior and sample from the observation for different tunnings