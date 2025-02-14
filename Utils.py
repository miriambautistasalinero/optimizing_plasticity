import numpy as np
import numba as nb
from numba import njit
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import torch



def gaussian(x, A, mu, sigma, R0):
    
    diff = np.mod(x - mu + np.pi/2, np.pi) - np.pi/2  

    # R0 is the baseline firing rate
    return R0 + np.abs(A) * np.exp(- diff**2 / (2 * sigma**2)) 


def get_tuning_stats(angle, r, pulse_time, dt, normalize_A = True):
    
    l = int(len(angle)//2)
    x = angle[-l:]
    y = np.mean(r.reshape(-1, int(pulse_time/dt)), axis=1)[-l:]
    
    if normalize_A:
        y = y/(np.max(y) + 0.001)
    
    if (np.isnan(y).any() or np.isinf(y).any()):
        A = 0
        mu = 0
        sigma = 100
        r_squared = 0
        R0 = 0
        
    else:
 
        # Fit the Gaussian
        try:
            popt, pcov = curve_fit(gaussian, x, y,
                                   p0=[np.max(y), x[y == np.max(y)][0], 0.3490, 0.0], maxfev = 150000)
            
            A, mu, sigma, R0 = popt
            
            # Calculate the R-squared value
            r_squared = r2_score(y, gaussian(x, *popt))
 
            
        except RuntimeError:
            
            print("Error - curve_fit failed")
            A = 0
            mu = 0
            sigma = 100
            r_squared = 0
            R0 = 0


    return A, mu, sigma, r_squared, R0


# Sets the tunning curve of the neurons 
def activation(x, y, sigma = np.pi/9):
    
    diff = np.mod(x - y + np.pi/2, np.pi) - np.pi/2  

    R0 = 0.01
    # R0 is the baseline firing rate
    # Should it be normalized?
    return R0 + np.exp(- diff**2 / (2 * sigma**2)) 

@njit
def non_lin(x, params, nonlin=True): # params=np.array([0.4, 60, 0.4, 1.3, -5, 0]
    if nonlin:
        offset = params[0] / (1 + np.exp(params[1] * (0 - params[2]) ** 2)) + params[3] / (1 + np.exp(params[4] * (0 -params[5])))
        y = params[0] / (1 + np.exp(params[1] * (x - params[2]) ** 2)) + params[3] / (1 + np.exp(params[4] * (x - params[5]))) - offset
        
    else:
        c = params[3]/(params[5]*2)
        y = c*x
    return y

@njit
def stdp(pre, post, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0):
    return alpha * pre * post + beta * pre + gamma * post + delta

@njit
def rate(x, params=np.array([1.0, 0.25, 2.0])):
    return params[0] * np.maximum(x - params[1], 0)**params[2]



def get_EI_input(noise = 0.1,
             E_inputs = 200,
             I_inputs = 50,
             Groups = 10,
             scale = 1,
             pulse_time = 100,
             time = 10000,
             dt = 0.1,
             I_delay = 20):
    
        timesteps = int(time/dt)
        pulse_len = int(pulse_time/dt)
        I_delay_steps = int(I_delay/dt)
        # number of time pulses
        num_pulse = int(time/pulse_time)
        # The angle of the bar at each pulse
        angle = np.pi*np.random.rand(num_pulse)

        # possible selectivity orientations
        selectivity = np.linspace(0, np.pi-(np.pi/Groups), Groups)
        
        E_group_size = int(E_inputs/Groups)
        # Each input group is selective to one orientation
        E_Selectivity = np.repeat(selectivity, E_group_size)
        Se = np.ones((timesteps, E_inputs))
        
        I_group_size = int(I_inputs/Groups)
        I_Selectivity = np.repeat(selectivity, I_group_size)
        Si = np.ones((timesteps, I_inputs))


        for i in range(num_pulse):
            Se[i*pulse_len:(i+1)*pulse_len, :] = activation(E_Selectivity, angle[i], sigma = np.pi/6)
            Si[i*pulse_len:(i+1)*pulse_len, :] = activation(I_Selectivity, angle[i], sigma = np.pi/4)

        # why scale it in this way?
        E_currents = (1-noise)*Se + 2*np.mean(Se)*noise*np.random.rand(timesteps, E_inputs)
        I_currents = (1-noise)*Si + 2*np.mean(Si)*noise*np.random.rand(timesteps, I_inputs)
        # Rolls the inhibitory input to a certain delay
        I_currents = np.roll(I_currents, I_delay_steps, axis = 0) 

        return scale*E_currents, scale*I_currents, angle


# Three different types of input
def get_input(input_type,
             noise = 0.9,
             Inputs = 100,
             Groups = 10,
             scale = 1,
             pulse_time = 5,
             time = 10000,
             dt = 0.1):
    
    timesteps = int(time/dt)

    if input_type =='Sinusoid':
        angle = 0
        E = []
        phase = np.linspace(0, 2*np.pi - 2*np.pi/Groups, Groups)[::-1]

        start = np.random.rand()*np.pi
        for g in range(Inputs):
            Ei = np.sin(np.linspace(start + phase[int(Groups*g/Inputs)],
                                    start + timesteps*np.pi/10 + phase[int(Groups*g/Inputs)] ,
                                    timesteps))
            E.append(Ei)

        E_currents = (1-noise)*np.transpose(np.array(E)) + 2*noise*np.random.rand(timesteps, Inputs)
        I_currents = (1-noise)*np.transpose(np.array(E)) + 2*noise*np.random.rand(timesteps, Inputs)

        
    elif input_type == 'Pulses':
        
        angle = 0
                
        pulse_len = int(pulse_time/dt)
        num_pulse = int(time/pulse_time)
        active_group = np.random.randint(0, Groups, num_pulse)
        
        group_size = int(Inputs/Groups)

        S = np.ones((timesteps, Inputs))


        for i in range(num_pulse):
            S[i*pulse_len:(i+1)*pulse_len, active_group[i]*group_size:(active_group[i]+1)*group_size] *= 5

        E_currents = (1-noise)*S + 2*noise*np.random.rand(timesteps, Inputs)
        I_currents = (1-noise)*S + 2*noise*np.random.rand(timesteps, Inputs)
        
        
        

    elif input_type == 'Gratings':
                
        pulse_len = int(pulse_time/dt)
        num_pulse = int(time/pulse_time)
        angle = np.pi*np.random.rand(num_pulse)


        selectivity = np.linspace(0, np.pi, Groups)
        group_size = int(Inputs/Groups)
        Selectivity = np.repeat(selectivity, group_size)


        Se = np.ones((timesteps, Inputs))
        Si = np.ones((timesteps, Inputs))


        for i in range(num_pulse):
            Se[i*pulse_len:(i+1)*pulse_len, :] = activation(Selectivity, angle[i], sigma = np.pi/6)
            Si[i*pulse_len:(i+1)*pulse_len, :] = activation(Selectivity, angle[i], sigma = np.pi/4)

        E_currents = (1-noise)*Se + 2*np.mean(Se)*noise*np.random.rand(timesteps, Inputs)
        I_currents = (1-noise)*Si + 2*np.mean(Si)*noise*np.random.rand(timesteps, Inputs)
        
    
    else: 
        raise Exception("Unrecognized Input")
    
    return scale*E_currents, scale*I_currents, angle

#@njit
def get_input_matrix(In, Out, r):
    W_mask = np.zeros((Out, In))
    
    Os = Out//In
       
    for s in range(In):
        
        W_mask[s*Os:(s+1)*Os, s] = 1
        
    #Shuffle
    for s in range(Out):
        if np.random.rand() < r:
            x = np.random.randint(Out)
            W_mask[[s, x]] =  W_mask[[x, s]]

    return W_mask

#@njit
def get_losses(w_e, w_i, timesteps, Groups = 10, dt = 0.1):
    
    half_timesteps = int(timesteps / 2)

    # Coefficient of Variation
    cv_we = np.zeros(w_e.shape[1])
    
    mean_activity_we = np.mean(w_e[-half_timesteps:], axis = 0)
    std_neuron_we = np.std(w_e[-half_timesteps:], axis = 0)

    for ne in range(w_e.shape[1]):
        if mean_activity_we[ne] != 0:
            cv_we[ne] = std_neuron_we[ne] / mean_activity_we[ne]
        else:
            cv_we[ne] = 0
    #cv_we = np.where(mean_activity_we != 0, std_neuron_we / mean_activity_we, 0 )
    #cv_we = torch.where(mean_activity_we != 0, std_neuron_we / mean_activity_we, torch.tensor(0.0))
    #print(cv_we)
    cv_wi = np.zeros(w_i.shape[1])
    mean_activity_wi = np.mean(w_i[-half_timesteps:], axis = 0)
    std_neuron_wi = np.std(w_i[-half_timesteps:], axis = 0)
    #cv_wi = np.where(mean_activity_wi != 0, std_neuron_wi / mean_activity_wi, 0 )
    for ni in range(w_i.shape[1]):
        if mean_activity_wi[ni] != 0:
            cv_wi[ni] = std_neuron_wi[ni] / mean_activity_wi[ni]
        else:
            cv_wi[ni] = 0
    
    cv_we_mean = np.mean(cv_we)
    cv_wi_mean = np.mean(cv_wi)

    CV_sum = cv_we_mean + cv_wi_mean
    
    # OLD WEIGHT STABILITY
    Weight_Stability_old = 1 - (np.mean(np.std(w_e[-half_timesteps:], axis = 0)) + np.mean(np.std(w_i[-half_timesteps:], axis = 0)))
    
    # POPULATION STABILITY
    # POPULATION STABILITY
    ps_we = np.std(w_e[-half_timesteps:], axis = 0)/np.mean(np.mean(w_e[-half_timesteps:], axis = 0))
    ps_wi = np.std(w_i[-half_timesteps:], axis = 0)/np.mean(np.mean(w_i[-half_timesteps:], axis = 0))

    Population_Stability = 1 - (np.mean(ps_we) + np.mean(ps_wi))

    # POPULATION MEAN
    # WINNER TAKES ALL
    neuron_mean_e = np.mean(w_e[-half_timesteps:], axis = 0)
    pm_we = np.mean(neuron_mean_e / np.max(neuron_mean_e))
    neuron_mean_i = np.mean(w_i[-half_timesteps:], axis = 0)
    pm_wi = np.mean(neuron_mean_i / np.max(neuron_mean_i))
   

    n_random_checks = 1000
    b = np.zeros(n_random_checks)
    d_e = np.zeros(n_random_checks)
    d_i = np.zeros(n_random_checks)

    random_steps = np.random.choice(np.arange(half_timesteps, timesteps), n_random_checks, replace = False)
    for ri, rs in enumerate(random_steps):
    # E/I weight Balance
        WE = np.reshape(w_e[rs], (Groups, len(w_e[-1])//Groups)) 
        WI = np.reshape(w_i[rs], (Groups, len(w_i[-1])//Groups))
        # Detailed Balance
        b[ri] = np.corrcoef(np.mean(WE, axis = 1), np.mean(WI, axis = 1))[0, 1]
        d_e[ri] = np.sum(np.std(WE, axis = 1))/(Groups * np.std(WE))
        d_i[ri] = np.sum(np.std(WI, axis = 1))/(Groups * np.std(WI))
    
    
    mean_balance = np.mean(b)
    
    # Diversity
    #D_e = np.sum(np.std(WE, axis = 1))/(Groups*np.std(WE))
    #D_i = np.sum(np.std(WI, axis = 1))/(Groups*np.std(WI))
    
    mean_diversity = 1 - (np.mean(d_e) + np.mean(d_i))

    # POPULATION STABILITY
    # POPULATION STABILITY
    ps_we = np.std(w_e[-half_timesteps:], axis = 0)/np.mean(np.mean(w_e[-half_timesteps:], axis = 0))
    ps_wi = np.std(w_i[-half_timesteps:], axis = 0)/np.mean(np.mean(w_i[-half_timesteps:], axis = 0))

    Population_Stability = 1 - (np.mean(ps_we) + np.mean(ps_wi))


    # Previous measures of balance and diversity
    WE_last = np.reshape(w_e[-1], (Groups, len(w_e[-1])//Groups))
    WI_last = np.reshape(w_i[-1], (Groups, len(w_i[-1])//Groups))

    Balance = np.corrcoef(np.mean(WE_last, axis = 1), np.mean(WI_last, axis = 1))[0, 1]

    D_e = np.sum(np.std(WE_last, axis = 1))/(Groups*np.std(WE_last))
    D_i = np.sum(np.std(WI_last, axis = 1))/(Groups*np.std(WI_last))
    
    Diversity = 1 -(D_e + D_i)

    window_length = 1000/dt #1s windows
    windows = np.arange(half_timesteps, timesteps, window_length, dtype = int)
    ld_e = np.zeros(len(windows)-1)
    ld_i = np.zeros(len(windows)-1)
    epsilon = 1e-10  # Small constant to avoid division by zero

    for tw in range(len(windows)-1):
        
        start_window = windows[tw]
        end_window = windows[tw+1]
       
        w_e_start = np.reshape(w_e[start_window], (Groups, len(w_e[-1])//Groups)) 
        w_e_end = np.reshape(w_e[end_window], (Groups, len(w_e[-1])//Groups))

        w_i_start = np.reshape(w_i[start_window], (Groups, len(w_i[-1])//Groups)) 
        w_i_end = np.reshape(w_i[end_window], (Groups, len(w_i[-1])//Groups))

        #print("w_e_start: ", w_e_start, "w_i_start: ", w_i_start)
        #print("w_e_end: ", w_e_end, "w_i_end: ", w_i_end)
        #print(w_e_start.mean(axis = 1))
        ld_e[tw] = np.sum(np.abs(w_e_end.mean(axis = 1) - w_e_start.mean(axis = 1)) / (w_e_start.mean(axis = 1) + w_e_end.mean(axis = 1)+ epsilon))
        ld_i[tw] = np.sum(np.abs(w_i_end.mean(axis = 1) - w_i_start.mean(axis = 1)) / (w_i_start.mean(axis = 1) + w_i_end.mean(axis = 1)+ epsilon))
        #print(ls_e[tw], ls_i[tw])
        tw +=2
        #start += 2
        #end += 2

    local_drift_e = np.mean(ld_e)
    local_drift_i = np.mean(ld_i)
    

    return Weight_Stability_old, Balance, Diversity, local_drift_e, local_drift_i, mean_balance, mean_diversity, Population_Stability, cv_we_mean, cv_wi_mean, CV_sum,  pm_we, pm_wi


## BASELINE EXPERIMENT

@njit
def simulate_neuron(dendrites, e_target, i_target, eta_e, eta_i,  nonlinparams, E_inputs, I_inputs, E_rates, I_rates, W_input_E, W_input_I, dendrite_leak, tau, dt, params, time):

    # Excitatory
    # Plasticity parameters
    alpha_e = 1.0 #  params[0] #
    beta_e = 0 # params[1] #0
    gamma_e = 0 #params[2] #
    delta_e = 0 #params[3] #

    # Inhibitory
    alpha_i =  params[0] 
    beta_i =  params[1] 
    gamma_i = params[2] 
    delta_i = params[3] 
    
    timesteps=int(time/dt)
    # initialize rate of postsynaptic neuron
    r = np.zeros(timesteps)
    # initialize voltage of postsynaptic neuron
    v = np.zeros(timesteps)
    
    # initial values at t = 0 
    r[0] = 0.01
    v = 0.25

    # initialize weights to the specific target 
    w_e = e_target * np.random.rand(timesteps, E_inputs) / E_inputs
    w_i = i_target * np.random.rand(timesteps, I_inputs) / I_inputs

    
    E_target = e_target * np.ones(timesteps)

    # Simulate learning
    for t in range(1, timesteps-1):
        # sum of inputs
        x = (w_e[t] * E_rates[t])@W_input_E - (w_i[t] * I_rates[t])@W_input_I + dendrite_leak * v
        #if dendrites == 0:
            #print("dentrites are false")
            # No dendritic non-linearity
        inp = np.sum(x) #* 6
        #else: 
            #inp = np.sum(non_lin(x, nonlin=True, params=nonlinparams))

        delta_v = (-v + inp) / tau
        v = v + dt * delta_v
    
        r[t] = rate(v)

        # STDP
        w_e[t+1] = np.maximum(w_e[t] + eta_e * dt * stdp(E_rates[t], r[t], alpha_e, beta_e, gamma_e, delta_e), 0) 
        w_i[t+1] = np.maximum(w_i[t] + eta_i * dt * stdp(I_rates[t], r[t], alpha_i, beta_i, gamma_i, delta_i), 0) 
  

        # Meta-Plasticity
        E_target[t+1] = E_target[t] - 0.25 * eta_e * dt * (r[t] - r[0])

        # Normalization
        w_e[t+1] = w_e[t+1] * E_target[t+1] / np.sum(w_e[t+1])
        w_i[t+1] = w_i[t+1] * i_target / np.sum(w_i[t+1])

    return w_e, w_i, r, E_target


def run_baseline_network(params, sbi_run=True):

   # Network Parameters
    E_inputs = 200
    I_inputs = 50
    Groups = 10
    # Total synaptic weight
    e_target = 5.0
    # Learning rate excitatory
    eta_e = 0.001
    # Learning rate inhibitory
    eta_i = 0.002
    # Total synaptic weight inhibitory
    i_target = 1.5

    # Simulation Parameters
    time = 100000 # ms
    dt = 0.1  # ms
    timesteps=int(time/dt)
    pulse_time = 50
    tau = 2.0 # ms
    noise = 0.1
    # Dendritic non-linearity
    dendrites = 0
    I_delay = 0
    scale = 0.5 #0.5 #
    dendrite_leak = 0
    nonlinparams = 0

    re = 0.03
    ri = 0.03
    # W_dend_E is (200,10), binary matrix that indicate which neuron belong to each group in a shuffled way np.sum(W_dend_E, axis=0) = array of 1x10 -
    W_input_E = get_input_matrix(Groups, E_inputs, re)
    W_input_I = get_input_matrix(Groups, I_inputs, ri)

    # Get Inputs
    E_rates, I_rates, angle = get_EI_input(noise = noise,
                                                E_inputs = E_inputs,
                                                I_inputs = I_inputs,
                                                Groups = Groups,
                                                scale = scale,
                                                pulse_time = pulse_time,
                                                time = time,
                                                dt = dt,
                                                I_delay =  I_delay)
            


    # ----- SIMULATION ----- #
    #                                       dendrites, e_target, i_target, eta_e, eta_i,  nonlinparams, E_inputs, I_inputs, E_rates, I_rates, W_input_E, W_input_I, dendrite_leak, tau, dt, params, time
    w_e, w_i, r, E_target = simulate_neuron(dendrites, e_target, i_target, eta_e, eta_i, nonlinparams, E_inputs, I_inputs, E_rates, I_rates, W_input_E, W_input_I, dendrite_leak, tau, dt, params, time)
    # --- LOSSES --- #
    S, B, D, LD_e, LD_i, mean_B, mean_D, Population_Stability, cv_we, cv_wi, CV_sum,  pm_we, pm_wi  = get_losses(w_e, w_i, timesteps)
    # --- TUNNING FUNCTION --- #
    A, mu, sigma, r_squared, R0 = get_tuning_stats(angle, r, pulse_time, dt)

    if sbi_run == False:
    # For plotting and analyzing the network
        return w_e, w_i, r, E_target, E_rates, I_rates, angle, S, B, D,  A, mu, sigma, r_squared, R0, mean_B, mean_D, LD_e, LD_i, Population_Stability, cv_we, cv_wi, CV_sum,  pm_we, pm_wi
    else:
        # for sbi
        return S, B, D, A, mu, sigma, r_squared, R0, LD_e, LD_i, mean_B, mean_D, Population_Stability, cv_we, cv_wi, CV_sum,  pm_we, pm_wi
    
def sigma_to_HWHLdegrees(x, sigma_index):
    # HWHL
    # sigma_index for sbi results is 5
    x[:, sigma_index] *= np.sqrt(2*np.log(2))
    # Degrees
    x[:, sigma_index] *= (180 / np.pi)
    return x


def percent_highervalues(data, percent):
    # Calculate the threshold value for the top 10%
    num_top = int(np.ceil(percent * len(data)))

    # Use np.partition to find the indices of the largest 10% values
    threshold_value = np.partition(data, -num_top)[-num_top]  # Find the smallest value in the top 10%
    indices_top_10_percent = np.where(data >= threshold_value)[0]



    return indices_top_10_percent
