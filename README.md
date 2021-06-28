# Continuous PSRL using GPs

## Infrastructure notes

Things we would like to add to the codebase:
* Add LBFGS optimisation
* Adaptive number of inducing points
* Add interfacing to environment's dynamics and rewards as models for ground truth training/testing
* Change initial distribution handling to prior-posterior setup
* Add rng to various classes: models, agents, policy
* Functionality for evaluating models on randomly sampled ground truth data (performance measures)
* Add oracle agent for low-dimensional environments
* Agent snapshot saving and loading
* Consider adding stateful optimiser

## Fixes to get the agent working

Modifications, sanity checks and fixes to get the basic agent working:
* Increase subsampling factor to 2 or 3, reduce the horizon
* Ensure models and policy train properly:
  - Train GP models via LBFGS
  - Check the noise level learnt by the models (if this is too high, it could make optimisation very difficult)
  - Track model free energy for convergence
  - Use more rollouts for optimising the policy
  - Track policy optimisation for convergence
  - Try near-zero initial state variance
  - Give lots of data to the models, try to optimise policy

## General ideas

* Updating the models using continual learning
* Multi-output GPs, correlated output models
