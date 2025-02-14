print("Running classifier")

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

#from sbi.inference import prepare_for_sbi
print("After all the imports")


## Basic SBI commands
# 1. Simulate plasticity rules
def simulator_neuron(parameters):
    p = parameters.cpu().detach().numpy()
    (S, B, D, A, mu, sigma, r_squared, R0) = run_baseline_network(p)
    return (S, B, D, A, mu, sigma, r_squared, R0)


# number of parameters
num_dim = 8  

# Define priors
prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

print("Here")
# Ensure compliance with sbi's requirements
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(simulator_neuron, prior, prior_returns_numpy)

# Consistency check after making ready for sbi
check_sbi_inputs(simulator, prior)
#simulator, prior = prepare_for_sbi(simulator, prior)

## -- END BASICS
#theta, x = simulate_for_sbi(simulator, prior, 1000, num_workers=10)

# Load the round 1 simulations before classifier
data = np.load('saved/sim/nb_r1_simforclas.npz')
print("Data loaded", data['theta'].shape)

# If the data is loaded but not simulated - Convert from numpy to torch
theta = torch.from_numpy(data['theta'])
x = torch.from_numpy(data['x'])

# Restrict the prior using a classifier
restriction_estimator = RestrictionEstimator(prior=prior)
restriction_estimator.append_simulations(theta, x)
print("About to train")
classifier = restriction_estimator.train()
restricted_prior = restriction_estimator.restrict_prior()

restricted_prior = restriction_estimator.restrict_prior()
samples = restricted_prior.sample((10000,))
#with open("saved/results/SBI_restrictedprior_BN.pkl", "wb") as f:
    #pickle.dump(restricted_prior, f)
#print("Restricted prior saved!")
# Generate the new theta and x from the restricted prior
#new_theta, new_x = simulate_for_sbi(simulator, restricted_prior, num_simulations=100,
                            #num_workers=2,
                            #show_progress_bar=True)




