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


data = np.load('saved/sim/P4_r2_fixed_exc_data_rsqrt90.npz')

print("Data loaded")    

# If the data is loaded but not simulated - Convert from numpy to torch
theta_2 = torch.from_numpy(data['theta'])
x_2 = torch.from_numpy(data['x'])
print(theta_2.shape)
print(x_2.shape)

# Check for NaN values
print(torch.sum(torch.isnan(theta_2)))
print(torch.sum(torch.isinf(theta_2)))

# Check for NaN and Inf values in 'x'
print(torch.sum(torch.isnan(x_2)))
print(torch.sum(torch.isinf(x_2)))

# Range of values of theta
u = theta_2.unique()
print("Parameter ranges from"  + str(u.min()) + " to " + str(u.max()))


## FILTERING


# P2: Filter on sqrt > 0.9
# Filter on r2 values
#r2_mask = x_2[:,6] >= 0.9
# 2. Take sigma to train the neural density estimator
#x_filtered = x_2[r2_mask][:,6].unsqueeze(1)
#theta_filtered = theta_2[r2_mask]

# P4: Filter on 10% of population stability
#ps_10_index = percent_highervalues(x_2[:,12], 0.1)
#x_filtered = x_2[ps_10_index][:,12].unsqueeze(1)
#theta_filtered = theta_2[ps_10_index]

# P1 r2 filter for population stability, balance and diversity

popstability_mask = x_2[:,12] > 0.9
#balance_mask = x_2[:,1] > 0.9
#diversity_mask = x_2[:,2] > 0.9

combined_mask = popstability_mask #& balance_mask & diversity_mask
x_filtered = x_2[combined_mask][:,12].unsqueeze(1)
theta_filtered = theta_2[combined_mask]


print("x_filtered", x_filtered.shape)
# 3. Train using sigma
inference_2 = SNPE(density_estimator="nsf")

# train the neural density estimator
density_estimator_2 = inference_2.append_simulations(theta_filtered, x_filtered).train()
#with open("saved/results/nb_r2_density_estimator_large_allmetrics_ftrainsbd95_ftrsqrt9.pkl", "wb") as f:
    #pickle.dump(density_estimator_2, f)


print("Before posterior")
# Generate the posterior
posterior_final = inference_2.build_posterior(density_estimator_2)

# Save the posterior
#np.savez(f'saved/results/SBI_posterior_plasticity_final.npz', posterior=posterior_final)
with open("saved/results/P4_r2_fixed_exc_data_rsqrt90_stabil90.pkl", "wb") as f:
    pickle.dump(posterior_final, f)
print("Second posterior saved")
