import mat73
import numpy as np
import pandas as pd
from scipy.io import savemat

data = mat73.loadmat('data/1_pilot/Bayesian_JAGS_model_selection_partial_pooling.mat')['samples']
n_chains, n_samples, n_participants = np.shape(data['z'])

z = data['z'] % 2
print(np.shape(z))

z_i = z.reshape(n_chains * n_samples, n_participants)

count_ones = np.sum(z_i, axis=0)
count_zeros = z_i.shape[0] - count_ones

# Create the new array with shape (2, 10) to store the counts
counts = np.array([count_zeros, count_ones])
counts += 1

# Calculate the proportions
total_counts = counts.sum(axis=0)
proportions = counts / total_counts.astype(float)

# Calculate the logarithm (base 10) of the proportions
log_proportions =  np.log10(proportions)


# Create a pandas DataFrame with the log_proportions array
df = pd.DataFrame(log_proportions[:2])

data_to_save = {'log_proportions': df.values}

# Save as a .mat file
savemat('log_proportions.mat', data_to_save)