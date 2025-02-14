import numpy as np


# Combine the results after all array jobs are completed (run this manually or via another script)
# You may use a separate script to aggregate all results:

all_theta = []
all_x = []
for i in range(10):
    print(i)
    data = np.load(f'new/saved/mini_sim/P4_r2_fixed_exc_data_rsqrt90_{i}.npz')
    #print(f"{i}, has {data["theta"].shape[0]}")
    all_theta.append(data['theta'])
    all_x.append(data['x'])

#for i in range(4,6):
#data = np.load(f'new/saved/sim/SBI_data_WA1_2HT_r2_uncomplete.npz')

#all_theta.append(data['theta'])
#all_x.append(data['x'])
    
#for i in range(7,10):
    #data = np.load(f'new/saved/mini_sim/SBI_data_WA1_2HT_r2_{i}.npz')

    #all_theta.append(data['theta'])
    #all_x.append(data['x'])

np.savez('new/saved/sim/P4_r2_fixed_exc_data_rsqrt90.npz', theta=np.concatenate(all_theta), x=np.concatenate(all_x))
#with open("new/saved/sim/SBI_data_r1_1HT_newprior14.pkl", "wb") as f:
    #pickle.dump({'theta': all_theta, 'x': all_x}, f)
print("simulations combined")

