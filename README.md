# Continuous PSRL using GPs

## Infrastructure notes

Things we would like to add to the codebase:
* Adaptive number of inducing points
* Model testing on simple functions
* Modify GPs to use deltas
* Add interfacing to environment's dynamics and rewards as models for ground truth training/testing.
* Change initial distribution handling to prior-posterior setup
* Add rng to various classes: models, agents, policy
* Functionality for testing models on randomly sampled ground truth data
* Agent snapshot saving and loading
* Initialisation helper (maybe in VFEGP stack?)


## General ideas

* Updating the models using continual learning
* Multi-output GPs, correlated output models
