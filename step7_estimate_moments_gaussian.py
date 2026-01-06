# %%
import numpy as np
import mat73
import yaml
from scipy.stats import gaussian_kde
# %%

def estimate_mode(data, session):
    data_tmp = data[:, :, session].ravel()
    kde = gaussian_kde(data_tmp)
    mode_val = data_tmp[np.argmax(kde.pdf(data_tmp))]
    mode_prob = kde.pdf(mode_val)

    return mode_val, mode_prob

config = "config_files/config_2_full.yaml"

with open(f"{config}", "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)


quality_dictionary = {'chains': [2,4,4,4], 'samples': [5e1,5e2,5e3,1e4,2e4], 'manual_burnin': [1e1,1e3,1e3,2e4,4e4]}
n_agents = config["n_agents"]
burn_in = int(quality_dictionary['manual_burnin'][config['qual'] - 1])


full_pooling = mat73.loadmat("data/2_full_data/Bayesian_JAGS_parameter_estimation_full_pooling.mat")

full_pooling_etas = full_pooling["samples"]["eta_g"][:, burn_in:, :]

add = 0
mult = 1

print("=====================================")
print("Full-pooling - eta_g: additive ========")
lower_bound = np.percentile(full_pooling_etas[:, :, add].ravel(), 2.5)
upper_bound = np.percentile(full_pooling_etas[:, :, add].ravel(), 97.5)
middle_bound = np.percentile(full_pooling_etas[:, :, add].ravel(), 50)
print(f"95% Equal-Tailed Interval: [{lower_bound} === {middle_bound} === {upper_bound}]")

print("")

print("Full-pooling - eta_g: multiplicative ==")
lower_bound = np.percentile(full_pooling_etas[:, :, mult].ravel(), 2.5)
upper_bound = np.percentile(full_pooling_etas[:, :, mult].ravel(), 97.5)
middle_bound = np.percentile(full_pooling_etas[:, :, mult].ravel(), 50)
print(f"95% Equal-Tailed Interval: [{lower_bound} === {middle_bound} === {upper_bound}]")


print("\n")

partial_pooling = mat73.loadmat("data/2_full_data/Bayesian_JAGS_parameter_estimation_partial_pooling.mat")

partial_pooling_etas = partial_pooling["samples"]["eta_g"][:, burn_in:, :]

add = 0
mult = 1

print("Partial-pooling - eta_g : additive ========")
lower_bound = np.percentile(partial_pooling_etas[:, :, add].ravel(), 2.5)
upper_bound = np.percentile(partial_pooling_etas[:, :, add].ravel(), 97.5)
middle_bound = np.percentile(partial_pooling_etas[:, :, add].ravel(), 50)
print(f"95% Equal-Tailed Interval: [{lower_bound} === {middle_bound} === {upper_bound}]")

print("")

print("Partial-pooling - eta_g: multiplicative ==")
lower_bound = np.percentile(partial_pooling_etas[:, :, mult].ravel(), 2.5)
upper_bound = np.percentile(partial_pooling_etas[:, :, mult].ravel(), 97.5)
middle_bound = np.percentile(partial_pooling_etas[:, :, mult].ravel(), 50)
print(f"95% Equal-Tailed Interval: [{lower_bound} === {middle_bound} === {upper_bound}]")



print("Mode estimation")

print("Partial-pooling - eta_g : additive ========")
m, p = estimate_mode(partial_pooling_etas[:, :, :], add)
print(f"mode based on kde - {m}, p = {p}")

print("")

print("Partial-pooling - eta_g: multiplicative ==")
m, p = estimate_mode(partial_pooling_etas[:, :, :], mult)
print(f"mode based on kde - {m}, p = {p}")