# Optimizing the emergence of orientation selectivity through meta-learned E/I plasticity rules

## Description
This repository contains Python code used to investigate a competitive Hebbian learning framework where excitatory and inhibitory plasticity are simultaneously adjusted. We aim to discover plasticity rules that drive the emergence of input selectivity using filter simulation-based inference, which sequentially narrows the rule search space. 

In order to explore the full range of possible rules, we used simulation based inference. We take as reference [^1], where simulation-based inference was used to infer the distribution of plasticity rules in spiking neural networks while ensuring strong biologically plausible constraints. 

All revelant files are inside the new/ repository

## Repository Structure

* **Utils.py**: File with the most relevant functions
  * *simulate_neuron* function that performs the mechanistic model simulation. Input weights are initialized and learning is simulated
  *  *run_baseline_network* Input weigths are initialized (*get_input_matrix*) and learning (*get_EI_input*) is simulated. The losses or metrics for each task are computed (*get_losses*) and the tuning curve statistics of the output neuron (*get_tuning_stats*)

As previously mentioned, for the current project we used filter sbi. For this we used the python based package for simulation-based inference [^2] The implementation goes as follows:
  * Prior Definition: Define a broad prior distribution from where the intiial parameters will be sampled.
  * Simulation: For each draw of parameters form the prior distribution the model is used to simulation the behaviour. Metrics for eacht ask are computed.
  * Filtering Protocol: A filtering protocol was defined sequentially. Only parameter
sets meeting the criteria progressed to the inference step.
* Density Estimation: Accepted rules were then used to train a density estimator,
capturing the relationship between the set of rules and the selected metrics.
* Posterior Calculation: The posterior distribution was computed based on either specific observation or a defined range, depending on the filtering metric chosen. This posterior distribution is used as prior for the next iterative step. 

To parallelize the implementation in a more efficient way and reduce computation cost each filtering round was separated in a python file. 
* *sbi_simulate_r1.py* : Defines the prior, ensures compliance of the model with sbi's requirements, simulated the network and saves the data.
*  *sbi_inference_r1.py* : Loads the saved data and converts to torch. Applies a filtering protocol. Performs the inference step and returns a posterior distribution that is saved.
*  *sbi_simulate_r2.py* : Uses the last posterior distribituion as prior to sample the new parameters from this.
*  The filtering protocol continues until all the rounds are completed

* **clasifier.py** : Taking as reference [^3] we tried to implement a classifier to reduce the amount of innacurate outputs from the mechanistic model. However, the amount of data we were simulating was already large enough and the used of the classfier did not improve the results.

**Python Notebooks**
The python botebooks found in the current repository where used to test the models and the implementation of sbi. 

# Plots
All the figures submited for the project were plotted in Jupyter Notebooks. The most relevents plots are the ones found in *P1_final_plots.ipynb*, *P2_final_plots.ipynb* and *P3_final_plots.ipynb*. 

### References
