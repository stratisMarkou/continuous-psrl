# Continuous PSRL using GPs

## Infrastructure notes

Things we would like to add to the codebase:
* Adaptive number of inducing points
* Test models on simple functions (unit tests, ensure model can recover correct parameters in variety of cases)
* Modify agent to predict state deltas
* Add interfacing to environment's dynamics and rewards as models for ground truth training/testing
* Change initial distribution handling to prior-posterior setup
* Add rng to various classes: models, agents, policy
* Functionality for evaluating models on randomly sampled ground truth data (performance measures)
* Add oracle agent for low-dimensional environments
* Agent snapshot saving and loading
* Consider adding stateful optimiser

## General ideas

* Updating the models using continual learning
* Multi-output GPs, correlated output models
