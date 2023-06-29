import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import plot_individual_heatmaps, read_Bayesian_output

cmap = plt.get_cmap("tab20")
colors = [cmap(i) for i in np.linspace(0, 1, 10)]

data_dir = "data/0_simulation/grid/"
fig_dir = "figs/0_simulation/grid/"
types = ['eta_n05', 'eta_00', 'eta_05', 'eta_10', 'eta_15', 'time_optimal']

for pooling in ['no_pooling', 'partial_pooling']:
    all_data_add = np.ones([len(types),10*5000*4])
    all_data_mul = np.ones([len(types),10*5000*4])
    for i, version in enumerate(types):
        data_dir = data_dir + version
        d = read_Bayesian_output(
                        os.path.join(data_dir, f"Bayesian_JAGS_parameter_estimation_{pooling}.mat")
                    )
        eta_i = d["eta_i"]
        eta_i_t = eta_i.transpose((2, 0, 1, 3))
        eta_i_t_r = np.reshape(eta_i_t, (10 * 5000 * 4, 2))
        all_data_add[i,:] = eta_i_t_r[:,0]
        all_data_mul[i,:] = eta_i_t_r[:,1]

    all_data = np.array([all_data_add.flatten(), all_data_mul.flatten()])

    h1 = plot_individual_heatmaps(all_data, colors, hue = np.repeat(np.arange(2), 5000*4*10))

    h1.savefig(os.path.join(fig_dir, f"grid_simulation_riskaversion_{pooling}.png"))