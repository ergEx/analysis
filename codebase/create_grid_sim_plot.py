import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import plot_individual_heatmaps, read_Bayesian_output

cmap = plt.get_cmap("tab20")


plt.rcParams.update({
    "text.usetex": True})
sns.set_context('paper', font_scale=1.1)

data_dir = "data/0_simulation/grid/"
fig_dir = "figs/0_simulation/grid/"
types = ['eta_n05', 'eta_00', 'eta_05', 'eta_10', 'eta_15', 'time_optimal']
colors = [cmap(i) for i in np.linspace(0, 1, len(types))]
#Bracketing
all_data_add = np.ones([len(types),10*5000*4])
all_data_mul = np.ones([len(types),10*5000*4])
for i, version in enumerate(types):
    data_dir_tmp = data_dir + version
    bracketing_overview = pd.read_csv(os.path.join(data_dir_tmp, "bracketing_overview.csv"), sep = '\t')

    df_tmp = bracketing_overview[bracketing_overview.participant != 'all']

    etas = np.ones([10,5000*4,2])
    for j, participant in enumerate(list(set(df_tmp.participant))):
        for c, con in enumerate(list(set(df_tmp.dynamic))):
            tmp_df = df_tmp.query('participant == @participant and dynamic == @con')
            if tmp_df.log_reg_std_dev.values <= 0:
                etas[j,:,c] if j > 0 else np.ones([5000*4])
                continue
            etas[j,:,c] = np.random.normal(tmp_df.log_reg_decision_boundary, tmp_df.log_reg_std_dev, 5000*4)
    etas_log_r = np.reshape(etas, (10 * 5000 * 4, 2))
    all_data_add[i,:] = etas_log_r[:,0]
    all_data_mul[i,:] = etas_log_r[:,1]

all_data = np.array([all_data_add.flatten(), all_data_mul.flatten()])
h1 = plot_individual_heatmaps(all_data.T, colors, hue = np.repeat(np.arange(len(types)), 5000*4*10),
                              limits=[-1.5, 2.0], x_fiducial=[0], y_fiducial=[1])
h1.savefig(os.path.join(fig_dir, f"grid_simulation_riskaversion_bracketing.pdf"), dpi=600, bbox_inches='tight')

#Bayesian
for pooling in ['no_pooling', 'partial_pooling']:
    all_data_add = np.ones([len(types),10*5000*4])
    all_data_mul = np.ones([len(types),10*5000*4])
    for i, version in enumerate(types):
        data_dir_tmp = data_dir + version
        d = read_Bayesian_output(
                        os.path.join(data_dir_tmp, f"Bayesian_JAGS_parameter_estimation_{pooling}.mat")
                    )
        eta_i = d["eta_i"]
        eta_i_t = eta_i.transpose((2, 0, 1, 3))
        eta_i_t_r = np.reshape(eta_i_t, (10 * 5000 * 4, 2))
        all_data_add[i,:] = eta_i_t_r[:,0]
        all_data_mul[i,:] = eta_i_t_r[:,1]

    all_data = np.array([all_data_add.flatten(), all_data_mul.flatten()])

    h1 = plot_individual_heatmaps(all_data.T, colors, hue = np.repeat(np.arange(len(types)), 5000*4*10),
                                  limits=[-1.5, 2.0],  x_fiducial=[0], y_fiducial=[1])

    h1.savefig(os.path.join(fig_dir, f"grid_simulation_riskaversion_{pooling}.pdf"), dpi=600, bbox_inches='tight')