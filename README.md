# Continuous PSRL using GPs

## Infrastructure notes

Things we would like to add to the codebase:
* Fix first step
* Finish agent implementation
* Modify GPs to use deltas
* Add rng to GPPSRL agent
* Functionality for testing models on randomly sampled ground truth data
* Agent snapshot saving and loading
* Initialisation helper (maybe in VFEGP stack?)
* Look into checkshape bug


## General ideas

* Updating the models using continual learning
* Multi-output GPs, correlated output models