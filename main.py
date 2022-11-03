import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import logistic_regression, read_hlm_output

simulation = True
RESET = 45
n_passive_runs = 3
root_path = os.path.dirname(__file__)
design_variant = 'two_gamble_new_c'
inference_mode = 'parameter_estimation'

condition_specs = {0.0:'Additive', 1.0:'Multiplicative'}

if simulation:
    active_phase_df = pd.read_csv(os.path.join(root_path,'data',design_variant,'simulations','all_active_phase_data.csv'), sep='\t')
    subjects = list(range(15))

    if not os.path.isfile(os.path.join(root_path,'data',design_variant,f'{inference_mode}_simulated_data.mat')):
        print('HLM mordel output not found!')
    else:
        HLM_samples = read_hlm_output(inference_mode = inference_mode, experiment_version = design_variant, dataSource = 'simulated_data')
    save_path = os.path.join(root_path, 'figs', design_variant, 'simulations')
else:
    passive_phase_df = pd.read_csv(os.path.join(root_path,'data',design_variant,'all_passive_phase_data.csv'), sep='\t')
    active_phase_df = pd.read_csv(os.path.join(root_path,'data',design_variant,'all_active_phase_data.csv'), sep='\t')
    subjects = set(passive_phase_df['participant_id'])

    if not os.path.isfile(os.path.join(root_path,'data',design_variant,f'{inference_mode}_real_data.mat')):
        print('HLM mordel output not found!')
    else:
        hlm_samples = read_hlm_output(inference_mode = inference_mode, experiment_version = design_variant, dataSource = 'real_data')

    save_path = os.path.join(root_path, 'figs', design_variant)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

logistic_regression_input = np.empty([len(subjects),7])



for i,subject in enumerate(subjects):
    print(i)
    logistic_regression_input[i,0] = subject
    fig, ax = plt.subplots(2,6, figsize=(20,7))
    fig.suptitle(f'Subject {subject}')
    for c,condition in enumerate(condition_specs.keys()):
        '''PASIVE PHASE'''
        if simulation:
            ax[c,0].plot()
        else:
            passive_subject_df = passive_phase_df.query('subject_id == @subject and eta == @condition').reset_index(drop=True)
            ax[c,0].plot(passive_subject_df.trial, passive_subject_df.wealth)
            for reset_idx in range(1,n_passive_runs):
                ax[c,1].axvline(x=RESET*reset_idx,color='grey',linestyle='--')
            ax[c,0].plot([], label="Reset",color='grey',linestyle='--')
            ax[c,0].legend()
        ax[c,0].set(title=f'Passive wealth',
            xlabel='Trial',
            ylabel='Wealth')

        '''ACTIVE PHASE'''
        if simulation:
            active_subject_df = active_phase_df.query('subject_id == @subject and eta == @condition').reset_index(drop=True)
        else:
            active_subject_df = active_phase_df.query('no_response != True and subject_id == @subject and eta == @condition').reset_index(drop=True)

        #Active trajectory
        ax[c,1].plot(active_subject_df.trial, active_subject_df.wealth)
        ax[c,1].set(title=f'Active wealth',
            xlabel='Trial',
            ylabel='Wealth')

        #Indifference eta plots
        plot_specs = {'color':{0:'orange', 1: 'b'}, 'sign':{0:'>', 1:'<'}}
        for ii, choice in enumerate(active_subject_df['selected_side_map']):
            trial = active_subject_df.loc[ii,:]
            if np.isnan(trial.indif_eta):
                continue

            ax[c,2].axvline(condition, linestyle='--', linewidth=1, color='k')

            ax[c,2].plot(trial.indif_eta, ii, marker=plot_specs['sign'][trial.min_max_sign], color=plot_specs['color'][trial.min_max_color])

            ax[c,2].set(title = f'Indifference eta',
                xlabel = 'Riskaversion ($\eta$)')

            ax[c,2].axes.yaxis.set_visible(False)

        #Choice probabilities in different indifference eta regions
        min_df = active_subject_df.query('min_max_sign == 0').reset_index(drop=True)
        max_df = active_subject_df.query('min_max_sign == 1').reset_index(drop=True)
        min, _ = np.histogram(min_df['indif_eta'], bins=[-np.inf, -0.5, 0, 1.0, 1.5, np.inf])
        max, _ = np.histogram(max_df['indif_eta'], bins=[-np.inf, -0.5, 0, 1.0, 1.5, np.inf])
        h = [max[i]/(max[i]+min[i]) for  i in range(len(min))]
        ticks = ['<-0.5','-1 - 0','0 - 1','1 - 1.5','>1.5']
        ax[c,3].bar(ticks,h)
        ax[c,3].set(title=f'Indif eta choice prob.')
        ax[c,3].tick_params(axis='x', labelrotation=45)

        #Indifference eta logistic regression

        df_tmp = active_subject_df.query('indif_eta.notnull()', engine='python')

        x_test, pred, ymin, ymax, idx_m, idx_l, idx_h = logistic_regression(df_tmp)


        ax[c,4].fill_between(x_test, ymin, ymax, where=ymax >= ymin,
                    facecolor='grey', interpolate=True, alpha=0.5,label='95 % confidence interval')

        ax[c,4].plot(x_test,pred,color='black')

        sns.regplot(x=np.array(df_tmp.indif_eta),
                    y=np.array(df_tmp.min_max_val),
                    fit_reg=False,
                    y_jitter=0.05,
                    ax=ax[c,4],
                    label='data',
                    color='grey',
                    scatter_kws={'alpha': 1, 's':20})

        ax[c,4].axhline(y=0.5,color='grey',linestyle='--')

        ax[c,4].set(title=f'Logistic regression',
            ylabel = '',
            xlabel = 'Indifference eta',
            yticks = [0, 0.5, 1],
            ylim = (-0.25, 1.25))

        logistic_regression_input[i,3*c+1] = x_test[idx_h]
        logistic_regression_input[i,3*c+2] = x_test[idx_m]
        logistic_regression_input[i,3*c+3] = x_test[idx_l]

        #HLM model
        #print(HLM_samples.keys())
        eta_dist = HLM_samples['eta'][:,:,i,c].flatten()
        print(np.shape(HLM_samples['eta']))
        sns.kdeplot(eta_dist,ax=ax[c,5])
        ax[c,5].set(title=f'HLM model')
        ax[c,5].axvline(condition, linestyle='--', linewidth=1, color='k')



    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'Subject {subject}'))

#
#logistic_regression_outout = pd.DataFrame(logistic_regression_input,
#                                          columns=['Subject',
#                                                   'Risk_aversion_estimate_Additive_low_conf',
#                                                   'Risk_aversion_estimate_Additive',
#                                                   'Risk_aversion_estimate_Additive_upper_conf',
#                                                   'Risk_aversion_estimate_Multiplicative_low_conf',
#                                                   'Risk_aversion_estimate_Multiplicative',
#                                                   'Risk_aversion_estimate_Multiplicative_upper_conf'])
#if simulation:
#    logistic_regression_outout.to_csv(os.path.join(root_path,'data',design_variant, 'simulations', 'logistic_regression_output.csv'), sep='\t')
#else:
#    logistic_regression_outout.to_csv(os.path.join(root_path,'data',design_variant, 'logistic_regression_output.csv'), sep='\t')
#
