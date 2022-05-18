import matplotlib.pyplot as plt
import os
from .utils.utils import read_active_data, plot_indifference_eta

root_path = os.path.join(os.path.dirname(__file__),)
n_subjects = 18
dynamics = ['Additive','Multiplicative']

#Non-parametric plot
left = [-50,-5]
fig, ax = plt.subplots(1,len(dynamics),figsize = (12,10))
for ii, session in enumerate(dynamics):
    choices,x1,x2,x3,x4,w,filenames = read_active_data(root_path=root_path, path_extension='data/all_data', session=session, wealth = 'current')
    axis = plot_indifference_eta(choices,x1,x2,x3,x4,w,filenames,ax[ii],dynamic=dynamics[ii], left = left[ii])
plt.savefig('../figs/indifference_eta_non_parametric.png')
#plt.show()

#Logistic regression from indifference etas



