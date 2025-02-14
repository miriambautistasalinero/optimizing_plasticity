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


data = np.load('saved/sim/nb_r3_data_complete_trainbd90_popstb10perc.npz')

print("Data loaded")    

# If the data is loaded but not simulated - Convert from numpy to torch
theta_3 = torch.from_numpy(data['theta'])
x_3 = torch.from_numpy(data['x'])
print(theta_3.shape)
print(x_3.shape)

# Check for NaN values
print(torch.sum(torch.isnan(theta_3)))
print(torch.sum(torch.isinf(theta_3)))

# Check for NaN and Inf values in 'x'
print(torch.sum(torch.isnan(x_3)))
print(torch.sum(torch.isinf(x_3)))

# Range of values of theta
u = theta_3.unique()
print("Parameter ranges from"  + str(u.min()) + " to " + str(u.max()))


## FILTERING

# Restrict sigma values from 0.1 to 0.7
#mask = [(x_2[:,5] >= 0.1) & (x_2[:,5] <= 0.7)]

# Filter on r2 values
#r2_mask = x_2[:,6] >= 0.9

# 2. Take sigma to train the neural density estimator
#x_filtered = x_2[r2_mask][:,6].unsqueeze(1)

# Filter for rsqrt > 0.75
rsqrt_mask = x_3[:,6] > 0.75
x_filtered = x_3[rsqrt_mask][:,6].unsqueeze(1)

theta_filtered = theta_3[rsqrt_mask]

print("theta_filtered", theta_filtered.shape)

print("x_filtered", x_filtered.shape)
# 3. Train using sigma
inference_3 = SNPE(density_estimator="nsf")

# train the neural density estimator
density_estimator_3 = inference_3.append_simulations(theta_filtered, x_filtered).train()
#with open("saved/results/nb_r2_density_estimator_large_allmetrics_ftrainsbd95_ftrsqrt9.pkl", "wb") as f:
    #pickle.dump(density_estimator_2, f)


print("Before posterior")
# Generate the posterior
posterior_final = inference_3.build_posterior(density_estimator_3)

# Save the posterior
#np.savez(f'saved/results/SBI_posterior_plasticity_final.npz', posterior=posterior_final)
with open("saved/results/nb_r3_posterior_complete_trainbd90_ps10higher_rsqrt.pkl", "wb") as f:
    pickle.dump(posterior_final, f)
print("Second posterior saved")
