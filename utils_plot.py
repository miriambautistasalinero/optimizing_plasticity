import numpy as np
from Utils import *
import matplotlib.pyplot as plt
from numba import njit
import seaborn as sns

import numba as nb
from tqdm import tqdm as tqdm


def plot_sample_baseline(params):

    # not necessary
    nonlinparams = np.array([0.0, 1.0, 1.0, 1.0, -600, 0.065])


    # Simulation Parameters
    E_inputs = 200
    I_inputs = 50
    Groups = 10


    time = 180000 # ms
    dt = 0.1 # ms
    timesteps=int(time/dt)
    pulse_time = 200 
    tau = 2.0
        
    noise = 0.2
    dendrite_leak = 0
    dendrites = False

    # Initialize weights 
    # Setting number of dendrites as the number of groups   
    num_dend = Groups

    re = 0.9999
    ri = 0.9999
    # W_dend_E is (200,10), binary matrix that indicate which neuron belong to each group in a shuffled way np.sum(W_dend_E, axis=0) = array of 1x10 -
    W_dend_E = get_input_matrix(num_dend, E_inputs, re)
    W_dend_I = get_input_matrix(num_dend, I_inputs, ri)

    # Plasticity Parameters
    # Excitatory   
    e_target = 5.0
    i_target = 1.5

    # Get Inputs
    E_currents, I_currents, angle = get_EI_input(noise = noise,
                                                E_inputs = E_inputs,
                                                I_inputs = I_inputs,
                                                Groups = Groups,
                                                scale = 0.5,
                                                pulse_time = pulse_time,
                                                time = time,
                                                dt = dt,
                                                I_delay = 0)
    # E_currents (timesteps, neurons) - each group of 10 neurons has the same activity
    # angle - (pulses,) - the orientation selected at each pulse 


    # CODE FROM UTILS
    # dendrites, e_target, i_target,  nonlinparams, E_inputs, I_inputs, E_currents, I_currents, W_input_E, W_input_I, dendrite_leak, tau, dt, params, time
    w_e, w_i, r, E_target = simulate_neuron(dendrites, e_target, i_target, nonlinparams, E_inputs, I_inputs, E_currents, I_currents, 
                                W_dend_E, W_dend_I, dendrite_leak, tau, dt, params, time)

    l = int(len(angle)//2)

    # Plot the sample
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize= (25, 7))


    plt.subplot(1, 3, 1)
    plt.plot(np.linspace(0, time, timesteps)[::2000], w_e[::2000], color = '#3d8bbeff', alpha = 0.8);
    plt.plot(np.linspace(0, time, timesteps)[::2000], -w_i[::2000]/4, color = '#d74c5eff', alpha = 0.8);
    plt.axhline(y = 0, linestyle = "--", linewidth = 5.0, color = 'k')
    plt.xticks([0, 60000, 120000], ['0', '1', '2'])
    plt.xlabel('Time (minutes)')
    plt.ylabel('Weights')
    plt.title('Weight Trajectories')
    sns.despine()

    plt.subplot(1, 3, 2)
    plt.plot(np.linspace(0, 100, E_inputs), w_e[-1], '.', markersize = 25, color = '#3d8bbeff', alpha = 0.5)
    plt.plot(np.linspace(0, 100, I_inputs), -w_i[-1]/4, '.', markersize = 25, color = '#d74c5eff', alpha = 0.5)
    plt.xlabel('Input Orientation')
    plt.xticks(np.linspace(0, 100, 4), ['0', r'$\pi/4$', r'$3\pi/4$', r'$\pi$'])
    plt.ylabel('Weights')
    plt.title('Learned Weights')


    plt.subplot(1, 3, 3)
    plt.plot(angle[-l:], np.mean(r.reshape(-1, int(pulse_time/dt)), axis=1)[-l:],
            'o',
            label='FR',
            color = '#688c7eca', alpha = 0.5)
    #plt.plot(x_fit, y_fit, '-', label='Gaussian', color = '#bc2d8fca', linewidth = 5)
    plt.xticks(np.linspace(0, np.pi, 4), ['0', r'$\pi/4$', r'$3\pi/4$', r'$\pi$'])
    plt.xlabel('Orientation')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Neuron Tuning Curve')
    plt.ylim([-0.0005, 0.1])
    sns.despine()

    plt.tight_layout()
    plt.show()


def samples_plot(w_e, w_i, r, angle):

    # Take the last half of the data - why? Because at the beginning the network is not stable
    l = int(len(angle)//2)
    # Network Parameters
    E_inputs = 200
    I_inputs = 50
    Groups = 10

    # Simulation Parameters
    time = 100000 # ms
    dt = 0.1  # ms
    timesteps=int(time/dt)
    pulse_time = 50

    #COMPUTE THE GAUSSIAN FIT WITHOUT NORMALIZATION
    A, mu, sigma, r_squared, R0 = get_tuning_stats(angle, r, pulse_time, dt, normalize_A = False)
    x_fit = np.linspace(0, np.pi, 1000)
    y_fit = gaussian(x_fit, A, mu, sigma, R0)

    
    plt.figure(figsize= (25, 7))


    plt.subplot(1, 3, 1)
    plt.plot(np.linspace(0, time, timesteps)[::2000], w_e[::2000], color = '#3d8bbeff', alpha = 0.8);
    plt.plot(np.linspace(0, time, timesteps)[::2000], -w_i[::2000]/4, color = '#d74c5eff', alpha = 0.8);
    plt.axhline(y = 0, linestyle = "--", linewidth = 5.0, color = 'k')
    plt.xticks(np.linspace(0, time, 3), ['0', '1', '2'])
    plt.xlabel('Time (minutes)')
    plt.ylabel('Weights')
    plt.title('Weight Trajectories')
    sns.despine()

    plt.subplot(1, 3, 2)
    plt.plot(np.linspace(0, 100, 200), w_e[-1], '.', markersize = 25, color = '#3d8bbeff', alpha = 0.5)
    plt.plot(np.linspace(0, 100, 50), -w_i[-1]/4, '.', markersize = 25, color = '#d74c5eff', alpha = 0.5)
    plt.xlabel('Input Orientation')
    plt.xticks(np.linspace(0, 100, 4), ['0', r'$\pi/4$', r'$3\pi/4$', r'$\pi$'])
    plt.ylabel('Weights')
    plt.title('Learned Weights')


    plt.subplot(1, 3, 3)
    plt.plot(angle[-l:], np.mean(r.reshape(-1, int(pulse_time/dt)), axis=1)[-l:],
            'o',
            label='FR',
            color = '#688c7eca', alpha = 0.5)
            
    plt.plot(x_fit, y_fit, '-', label='Gaussian', color = '#bc2d8fca', linewidth = 5)
    plt.xticks(np.linspace(0, np.pi, 4), ['0', r'$\pi/4$', r'$3\pi/4$', r'$\pi$'])
    plt.xlabel('Orientation')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Neuron Tuning Curve')
    sns.despine()

    plt.tight_layout()
    plt.show()