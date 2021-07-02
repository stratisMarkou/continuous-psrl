# Continuous PSRL using GPs

## Infrastructure notes

Things we would like to add to the codebase:
* __FIX ACTION BUG__
* Unit tests for initial distribution
* Ensure prior initial distribution is sensible
* Add LBFGS optimisation
* Add interfacing to environment's dynamics and rewards as models for ground truth training/testing
* Add rng to various classes: models, agents, policy
* Add oracle agent for low-dimensional environments
* Agent snapshot saving and loading
* Consider adding stateful optimiser

## Fixes to get the agent working

Modifications, sanity checks and fixes to get the basic agent working:
* Ensure models and policy train properly:
  - Train GP models via LBFGS
  - Give lots of data to the models, try to optimise policy

## General ideas

* Updating the models using continual learning
* Multi-output GPs, correlated output models
