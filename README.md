# analysis

## To do:
- [x] Create script that reads data
  - [x] csv
  - [x] .mat
  
### Indifference eta

#### non-parametric
- [x] make indifference eta plots
- [x] make wealth plots

#### Logitstic regression
- [X] Create framework to get eta estimates
- [X] Create framework to plot distributions
- [x] Update above frameworks to work on data from experiment
- [x] Create JASP file with t-test on difference


### Bayesian Models
- [x] Update runHLM1.m
- [x] Update setHLM.m
- [x] Update computeHLM.m


#### Step 3
- [x] Update JAGS script for parameter estimation
- [ ] Check parameter estimation runs
- [ ] Create function to read data into np.arrays
- [X] Create framework to get MAP estimates
- [X] Create framework to plot distributions
- [ ] Split plots into seperate for each parameter for each condition
  - [ ] eta: for eta_mul add etaM to eta
  - [ ] etaM: Mainly a check that it is 0 for additive
  - [ ] beta
- [ ] Create JASP file with t-test on etaM

#### Step 4
- [ ] Check model selection from HLM (step 4) runs
- [ ] Decide on how to visualize the results (Variational Bayesian analysis toolbox?) 
