"""
Created on Wed May  2 10:07:24 2018

@author: davidm

Edited on Mon March 21 2022

@author: Benjamin Skjold
"""
import numpy as np
import os, scipy.io
import pandas as pd
from utils.utils import retrieve_keypress

# Define data path for original data files
datapath = os.path.join(os.path.dirname(__file__),'data')
txtAppnd = {0: 'add', 1: 'mul'}
lambds = [0,1]

n_dynamics = len(txtAppnd)
n_subjects = 19

redo = {0: 1.0, 1: 100.0}# The growth factors in the txt files are multiplied with a
# factor of 100 (e.g. 183.02 instead of 1.8302), this list will be used to correct
# for this
for src in ['all_data','optimal_eta_trials_deleted']:
    datadict = {}
    for lambd in lambds:
        datapaths_txt = os.path.join(datapath,src,txtAppnd[lambd])
        for i in range(n_subjects):
            subjID = str(i+1)
            with open(os.path.join(datapaths_txt, f'{subjID}_2.txt'),'rb') as f: #2 means active phase
                df = pd.read_csv(f,sep='\t',header=0)

            '''
            Append which session first
            '''
            if lambd == 0: #only want to add information once
                if i < 9: #First 9 subjects had multiplicative session on day 1
                    datadict.setdefault('MultiplicativeSessionFirst',[]).append(1)
                else:
                    datadict.setdefault('MultiplicativeSessionFirst',[]).append(0)

            '''
            Retrieve growth rates, wealth change values and wealth and add to dictionary
            '''
            datadict.setdefault('gr1_1_'+txtAppnd[lambd],[]).append(np.array(df['gr1_1'])) #Growth rates
            datadict.setdefault('gr1_2_'+txtAppnd[lambd],[]).append(np.array(df['gr1_2']))
            datadict.setdefault('gr2_1_'+txtAppnd[lambd],[]).append(np.array(df['gr2_1']))
            datadict.setdefault('gr2_2_'+txtAppnd[lambd],[]).append(np.array(df['gr2_2']))
            datadict.setdefault('x1_'+txtAppnd[lambd],[]).append(np.array(df['x1'])) #Wealth change constant wealth
            datadict.setdefault('x2_'+txtAppnd[lambd],[]).append(np.array(df['x2']))
            datadict.setdefault('x3_'+txtAppnd[lambd],[]).append(np.array(df['x3']))
            datadict.setdefault('x4_'+txtAppnd[lambd],[]).append(np.array(df['x4']))
            datadict.setdefault('dx1_'+txtAppnd[lambd],[]).append(np.array(df['dx1']))     #Wealth change updating wealth
            datadict.setdefault('dx2_'+txtAppnd[lambd],[]).append(np.array(df['dx2']))
            datadict.setdefault('dx3_'+txtAppnd[lambd],[]).append(np.array(df['dx3']))
            datadict.setdefault('dx4_'+txtAppnd[lambd],[]).append(np.array(df['dx4']))
            datadict.setdefault('wealth_'+txtAppnd[lambd],[]).append(df['earnings'][0]) #Wealth = starting wealth
            datadict.setdefault('wealth_current_'+txtAppnd[lambd],[]).append(np.array(df['wealth'])) #Updating wealth

            '''
            Retrieve keypresses and recode into accept/reject left side gamble
            '''
            df, sublst = retrieve_keypress(df) #(1=chosen gamble on the left, 0= chosen gamble on the right)

            datadict.setdefault('choice_'+txtAppnd[lambd],[]).append(sublst)

    scipy.io.savemat(os.path.join(os.getcwd(),'data',f'CPH_data_{src}.mat'),datadict,oned_as='row')
    np.savez(os.path.join(os.getcwd(),'data',f'CPH_data_{src}.npz'),datadict = datadict)