import os

import mat73
import numpy as np

version = 'two_gamble_new_c'
mat = mat73.loadmat(os.path.join(os.path.dirname(__file__),'data', version, 'parameter_estimation_simulated_data.mat'))
print(type(mat))
print(mat['samples'].keys())
print(type(mat['samples']['beta']))
print(np.shape(mat['samples']['beta']))
