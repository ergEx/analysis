import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import logistic_regression

RESET = 45
n_passive_runs = 3
root_path = os.path.dirname(__file__)
design_variant = 'test'

condition_specs = {0.0:'Additive', 1.0:'Multiplicative'}

passive_phase_df = pd.read_csv(os.path.join(root_path,'data','experiment_output',design_variant,'all_passive_phase_data.csv'), sep='\t')
active_phase_df = pd.read_csv(os.path.join(root_path,'data','experiment_output',design_variant,'all_active_phase_data.csv'), sep='\t')

logistic_regression_input = np.empty([len(set(passive_phase_df['participant_id'])),7])

save_path = os.path.join(root_path, 'figs', design_variant)
if not os.path.isdir(save_path):
    os.makedirs(save_path)

for i,subject in enumerate(set(passive_phase_df['participant_id'])):
    logistic_regression_input[i,0] = subject
    fig, ax = plt.subplots(2,9, figsize=(20,7))
    fig.suptitle('Subject {subject}')
    for c,condition in enumerate(condition_specs.keys()):
        ax[c,0].text(1, 3, f'{condition_specs[condition]}',fontsize = 14, rotation=45)
        ax[c,0].set(xlim =(0, 8), ylim =(0, 8))
        ax[c,0].axis('off')

        '''PASIVE PHASE'''
        passive_subject_df = passive_phase_df.query('participant_id == @subject and eta == @condition').reset_index(drop=True)
        ax[c,1].plot(passive_subject_df.trial, passive_subject_df.wealth)
        ax[c,1].set(title=f'Passive wealth',
               xlabel='Trial',
               ylabel='Wealth')
        for reset_idx in range(1,n_passive_runs):
            ax[c,1].axvline(x=RESET*reset_idx,color='grey',linestyle='--')
        ax[c,1].plot([], label="Reset",color='grey',linestyle='--')
        ax[c,1].legend()

        '''ACTIVE PHASE'''
        active_subject_df = active_phase_df.query('no_response != True and participant_id == @subject and eta == @condition').reset_index(drop=True)
        active_subject_df['EV1_2'] = active_phase_df[['x1_1','x1_2']].mean(axis=1)
        active_subject_df['EV3_4'] = active_phase_df[['x2_1','x2_2']].mean(axis=1)
        active_subject_df['dEV'] = active_subject_df['EV1_2'].sub(active_subject_df['EV3_4'], axis = 0).abs()
        active_subject_df['var1_2'] = active_phase_df[['x1_1','x1_2']].var(axis=1,ddof=0)
        active_subject_df['var3_4'] = active_phase_df[['x2_1','x2_2']].var(axis=1,ddof=0)
        active_subject_df['dvar'] = active_subject_df['var1_2'].sub(active_subject_df['var3_4'], axis = 0).abs()
        tmp = [1 if (active_subject_df.var1_2[i] > active_subject_df.var3_4[i] and active_subject_df.selected_side_map[i] == 0
                  or active_subject_df.var1_2[i] < active_subject_df.var3_4[i] and active_subject_df.selected_side_map[i] == 1) else 0 for i in range(len(active_subject_df))]
        active_subject_df['choose_high_variance'] = tmp

        #Active trajectory
        ax[c,2].plot(active_subject_df.trial, active_subject_df.wealth)
        ax[c,2].set(title=f'Active wealth',
               xlabel='Trial',
               ylabel='Wealth')

        #Additive choice probabilities
        '''NOT IMPLEMENTED'''
        ax[c,3].plot()
        ax[c,3].set(title=f'Add choice prob')

        #Multiplicative choice probabilities
        '''Not IMPLEMENTED'''
        ax[c,4].plot()
        ax[c,4].set(title=f'Mul choice prob')

        #Indifference eta plots
        for ii, choice in enumerate(active_subject_df['selected_side_map']):
            trial = active_subject_df.loc[ii,:]
            if np.isnan(trial.indif_eta):
                continue

            ax[c,5].axvline(condition, linestyle='--', linewidth=1, color='k')

            ax[c,5].plot(trial.indif_eta, ii, marker=trial.min_max_sign, color=trial.min_max_color)

            ax[c,5].set(title = f'Indifference eta',
                   xlabel = 'Riskaversion ($\eta$)')

            ax[c,5].axes.yaxis.set_visible(False)

        #Choice probabilities in different indifference eta regions
        ax[c,6].plot()
        ax[c,6].set(title=f'Indif eta choice prob.')

        #Indifference eta logistic regression

        df_tmp = active_subject_df.query('indif_eta.notnull()', engine='python')

        x_test, pred, ymin, ymax, idx_m, idx_l, idx_h = logistic_regression(df_tmp)


        ax[c,7].fill_between(x_test, ymin, ymax, where=ymax >= ymin,
                    facecolor='grey', interpolate=True, alpha=0.5,label='95 % confidence interval')

        ax[c,7].plot(x_test,pred,color='black')

        sns.regplot(x=np.array(df_tmp.indif_eta),
                    y=np.array(df_tmp.min_max_val),
                    fit_reg=False,
                    y_jitter=0.05,
                    ax=ax[c,7],
                    label='data',
                    color='grey',
                    scatter_kws={'alpha': 1, 's':20})

        ax[c,7].axhline(y=0.5,color='grey',linestyle='--')

        ax[c,7].set(title=f'Logistic regression',
               ylabel = '',
               xlabel = 'Indifference eta',
               yticks = [0, 0.5, 1],
               ylim = (-0.25, 1.25))

        logistic_regression_input[i,3*c+1] = x_test[idx_h]
        logistic_regression_input[i,3*c+2] = x_test[idx_m]
        logistic_regression_input[i,3*c+3] = x_test[idx_l]

        #HLM model
        '''NOT IMPLEMENTED'''
        ax[c,8].plot()
        ax[c,8].set(title=f'HLM model')
        if not os.path.isfile(os.path.join(root_path,'data','experiment_output',design_variant,'bayesian_model_output')):
            print('HLM mordel output not found!')
            continue

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'Subject {subject}'))


logistic_regression_outout = pd.DataFrame(logistic_regression_input,
                                          columns=['Subject',
                                                   'Risk_aversion_estimate_Additive_low_conf',
                                                   'Risk_aversion_estimate_Additive',
                                                   'Risk_aversion_estimate_Additive_upper_conf',
                                                   'Risk_aversion_estimate_Multiplicative_low_conf',
                                                   'Risk_aversion_estimate_Multiplicative',
                                                   'Risk_aversion_estimate_Multiplicative_upper_conf'])
logistic_regression_outout.to_csv(os.path.join(root_path,'data','experiment_output',design_variant, 'logistic_regression_output.csv'), sep='\t')
