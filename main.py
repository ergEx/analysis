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
active_limits = {0.0: [-500, 2_500], 1.0: [64 , 15_589]}

if simulation:
    active_phase_df = pd.read_csv(os.path.join(root_path,'data',design_variant,'simulations','all_active_phase_data.csv'), sep='\t')
    subjects = ['0.0x0.0','0.0x0.5','0.0x1.0','0.5x0.0','0.5x0.5','0.5x1.0','1.0x0.0','1.0x0.5','1.0x1.0']

    if not os.path.isfile(os.path.join(root_path,'data',design_variant,f'{inference_mode}_simulated_data.mat')):
        print('HLM model output not found!')
        hlm_samples_found = False
    else:
        HLM_samples = read_hlm_output(inference_mode = inference_mode, experiment_version = design_variant, dataSource = 'simulated_data')
        hlm_samples_found = True
    save_path = os.path.join(root_path, 'figs', design_variant, 'simulations')
else:
    passive_phase_df = pd.read_csv(os.path.join(root_path,'data',design_variant,'all_passive_phase_data.csv'), sep='\t')
    active_phase_df = pd.read_csv(os.path.join(root_path,'data',design_variant,'all_active_phase_data.csv'), sep='\t')
    subjects = set(passive_phase_df['participant_id'])

    if not os.path.isfile(os.path.join(root_path,'data',design_variant,f'{inference_mode}_real_data.mat')):
        print('HLM mordel output not found!')
        hlm_samples_found = False
    else:
        HLM_samples = read_hlm_output(inference_mode = inference_mode, experiment_version = design_variant, dataSource = 'real_data')
        hlm_samples_found = True

    save_path = os.path.join(root_path, 'figs', design_variant)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

logistic_regression_input = np.empty([len(subjects),7])

full_data = {0.0: {'indif_eta': [], 'min_max_sign': [], 'min_max_color':[], 'min_max_val':[]},
             1.0: {'indif_eta': [], 'min_max_sign': [], 'min_max_color':[], 'min_max_val':[]}}



for i,subject in enumerate(subjects):
    print(f'Subject {i+1} of {len(subjects)}')
    logistic_regression_input[i,0] = i
    fig, ax = plt.subplots(2,6, figsize=(20,7))
    fig.suptitle(f'Subject {subject}')
    for c,condition in enumerate(condition_specs.keys()):
        print(f'Condition {c+1} of {len(condition_specs)}')
        '''PASIVE PHASE'''
        if simulation:
            ax[c,0].plot()
        else:
            passive_subject_df = passive_phase_df.query('participant_id == @subject and eta == @condition').reset_index(drop=True)
            ax[c,0].plot(passive_subject_df.trial, passive_subject_df.wealth)
            for reset_idx in range(1,n_passive_runs):
                ax[c,0].axvline(x=RESET*reset_idx,color='grey',linestyle='--')
            ax[c,0].plot([], label="Reset",color='grey',linestyle='--')
            ax[c,0].legend(loc='upper left', fontsize='xx-small')
        ax[c,0].set(title=f'Passive wealth',
            xlabel='Trial',
            ylabel=f'Wealth')
        if c == 1:
            ax[c,0].set_yscale('log')

        '''ACTIVE PHASE'''
        if simulation:
            active_subject_df = active_phase_df.query('agent == @subject and eta == @condition').reset_index(drop=True)
            simulation_eta = float(subject.split('x')[c])
        else:
            active_subject_df = active_phase_df.query('no_response != True and subject_id == @subject and eta == @condition').reset_index(drop=True)

        #Active trajectory
        ax[c,1].plot(active_subject_df.trial, active_subject_df.wealth)
        ax[c,1].set(title=f'Active wealth',
            xlabel='Trial',
            ylabel='Wealth')

        ax[c,1].axhline(y=active_limits[condition][0], linestyle='--', linewidth=1,color='red', label='Upper Bound')
        ax[c,1].axhline(y=1000, linestyle='--', color='black', label='Starting Wealth')
        ax[c,1].axhline(y=active_limits[condition][1], linestyle='--', linewidth=1, color='red', label='Lower Bound')
        ax[c,1].legend(loc='upper left', fontsize='xx-small')
        if c == 1:
            ax[c,1].set(yscale='log',
                        ylabel='Wealth (log)')

        #Indifference eta plots
        plot_specs = {'color':{0:'orange', 1: 'b'}, 'sign':{0:'>', 1:'<'}}
        for ii, choice in enumerate(active_subject_df['selected_side_map']):
            trial = active_subject_df.loc[ii,:]
            if np.isnan(trial.indif_eta):
                continue
            full_data[condition]['indif_eta'].append(trial.indif_eta)
            full_data[condition]['min_max_sign'].append(trial.min_max_sign)
            full_data[condition]['min_max_color'].append(trial.min_max_color)
            full_data[condition]['min_max_val'].append(trial.min_max_val)
            ax[c,2].plot(trial.indif_eta, ii, marker=plot_specs['sign'][trial.min_max_sign], color=plot_specs['color'][trial.min_max_color])
        ax[c,2].set(title = f'Indifference eta',
                    xlabel = 'Riskaversion ($\eta$)')
        ax[c,2].axes.yaxis.set_visible(False)
        ax[c,2].axvline(condition, linestyle='--', color='grey', label='Growth optimal')
        if simulation:
            ax[c,2].axvline(simulation_eta, linestyle='--', color='green', label = 'Simulation eta')

        ax[c,2].scatter(1.2,10,marker='<',color='b', label='Upper bound')
        ax[c,2].scatter(-1.2,10,marker='>',color='orange', label='Lower bound')

        ax[c,2].legend(loc='upper left', fontsize='xx-small')

        #Choice probabilities in different indifference eta regions
        min_df = active_subject_df.query('min_max_sign == 0').reset_index(drop=True)
        max_df = active_subject_df.query('min_max_sign == 1').reset_index(drop=True)
        min, _ = np.histogram(min_df['indif_eta'], bins=[-np.inf, -0.5, 0, 1.0, 1.5, np.inf])
        max, _ = np.histogram(max_df['indif_eta'], bins=[-np.inf, -0.5, 0, 1.0, 1.5, np.inf])
        h = [max[i]/(max[i]+min[i]) for  i in range(len(min))]
        ticks = ['<-0.5','-1 - 0','0 - 1','1 - 1.5','>1.5']
        ax[c,3].bar(ticks,h)
        ax[c,3].set(title=f'Indif eta choice prob.',
                    ylim=[0,1],
                    yticks=np.linspace(0,1,11))
        ax[c,3].tick_params(axis='x', labelrotation=45)

        #Indifference eta logistic regression

        df_tmp = active_subject_df.query('indif_eta.notnull()', engine='python')
        df_tmp_1 = df_tmp[df_tmp['min_max_val'] == 1]
        df_tmp_0 = df_tmp[df_tmp['min_max_val'] == 0]

        print(f'Number og relevant gambles: {len(df_tmp) / len(active_subject_df):.2f}')
        x_test, pred, ymin, ymax, idx_m, idx_l, idx_h = logistic_regression(df_tmp)


        ax[c,4].fill_between(x_test, ymin, ymax, where=ymax >= ymin,
                    facecolor='grey', interpolate=True, alpha=0.5,label='95 % CI')

        ax[c,4].plot(x_test,pred,color='black')

        sns.regplot(x=np.array(df_tmp_1.indif_eta),
                    y=np.array(df_tmp_1.min_max_val),
                    fit_reg=False,
                    y_jitter=0.05,
                    ax=ax[c,4],
                    label='Upper Bound',
                    color='b',
                    marker='<',
                    scatter_kws={'alpha': 1, 's':20})
        sns.regplot(x=np.array(df_tmp_0.indif_eta),
                    y=np.array(df_tmp_0.min_max_val),
                    fit_reg=False,
                    y_jitter=0.05,
                    ax=ax[c,4],
                    label='Lower Bound',
                    color='orange',
                    marker='>',
                    scatter_kws={'alpha': 1, 's':20})

        ax[c,4].axvline(condition, linestyle='--', color='grey', label='Growth optimal')
        ax[c,4].axhline(y=0.5,color='grey',linestyle='--')



        ax[c,4].set(title=f'Logistic regression',
            ylabel = '',
            xlabel = 'Indifference eta',
            yticks = [0, 0.5, 1],
            ylim = (-0.25, 1.25),
            xticks = np.linspace(-5,5,11),
            xlim = [-5,5])

        logistic_regression_input[i,3*c+1] = x_test[idx_h]
        logistic_regression_input[i,3*c+2] = x_test[idx_m]
        logistic_regression_input[i,3*c+3] = x_test[idx_l]
        if simulation:
            ax[c,4].axvline(simulation_eta, linestyle='--', color='green', label='Simulation eta')
        ax[c,4].axvline(x=x_test[idx_m], color='red', linestyle='--', label = 'Best estimate')
        ax[c,4].legend(loc='upper left', fontsize='xx-small')

        #HLM model
        ax[c,5].axvline(condition, linestyle='--', color='grey', label='Growth optimal')
        if hlm_samples_found and i < 3:
            eta_dist = HLM_samples['eta'][:,:,i,c].flatten()
            sns.kdeplot(eta_dist,ax=ax[c,5])
            xs, ys = ax[c,5].lines[-1].get_data()
            ax[c,5].fill_between(xs, ys, color='red', alpha=0.05)
            mode_idx = np.argmax(ys)
            ax[c,5].vlines(xs[mode_idx], 0, ys[mode_idx], ls='--', color='red', label='Prediction')
            ax[c,5].axvline(condition, linestyle='--', linewidth=1, color='k')
        ax[c,5].set(title=f'Bayesian Model',
                ylabel = '',
                xticks = np.linspace(-5,5,11),
                xlim = [-5,5],
                xlabel='Risk aversion estimate')

        if simulation:
            ax[c,5].axvline(simulation_eta, linestyle='--', color='green', label='Simulation eta')

        ax[c,5].legend(loc='upper left', fontsize='xx-small')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'Subject {subject}'))


logistic_regression_outout = pd.DataFrame(logistic_regression_input,
                                          columns=['Subject',
                                                   'Additive_low_conf',
                                                   'Additive',
                                                   'Additive_upper_conf',
                                                   'Multiplicative_low_conf',
                                                   'Multiplicative',
                                                   'Multiplicative_upper_conf'])
if simulation:
    logistic_regression_outout.to_csv(os.path.join(root_path,'data',design_variant, 'simulations', 'logistic_regression_output.csv'), sep='\t')
else:
    logistic_regression_outout.to_csv(os.path.join(root_path,'data',design_variant, 'logistic_regression_output.csv'), sep='\t')


#### TREAT ALL AGENTS AS 1

fig, ax = plt.subplots(2,3, figsize=(20,12))
for c,condition in enumerate(condition_specs.keys()):
    df = pd.DataFrame.from_dict(full_data[condition])

    #Indifference eta plots
    plot_specs = {'color':{0:'orange', 1: 'b'}, 'sign':{0:'>', 1:'<'}}
    for ii in range(len(df)):
        trial = df.loc[ii,:]
        ax[c,0].plot(trial.indif_eta, ii, marker=plot_specs['sign'][trial.min_max_sign], color=plot_specs['color'][trial.min_max_color])
        ax[c,0].set(title = f'Indifference eta',
                    xlabel = 'Riskaversion ($\eta$)',
                    xlim=[-5,5])
    ax[c,0].axes.yaxis.set_visible(False)
    ax[c,0].axvline(condition, linestyle='--', color='grey', label='Growth optimal')
    ax[c,0].scatter(1.2,10,marker='<',color='b', label='Upper bound')
    ax[c,0].scatter(-1.2,10,marker='>',color='orange', label='Lower bound')

    ax[c,0].legend(loc='upper left')

    df_tmp = df.query('indif_eta.notnull()', engine='python')
    df_tmp_1 = df_tmp[df_tmp['min_max_val'] == 1]
    df_tmp_0 = df_tmp[df_tmp['min_max_val'] == 0]
    print(f'Number og relevant gambles: {len(df_tmp) / len(df):.2f}')

    x_test, pred, ymin, ymax, idx_m, idx_l, idx_h = logistic_regression(df_tmp)


    ax[c,1].fill_between(x_test, ymin, ymax, where=ymax >= ymin,
                facecolor='grey', interpolate=True, alpha=0.5)

    ax[c,1].plot(x_test,pred,color='black')

    sns.regplot(x=np.array(df_tmp_1.indif_eta),
                y=np.array(df_tmp_1.min_max_val),
                fit_reg=False,
                y_jitter=0.05,
                ax=ax[c,1],
                label='Upper Bound',
                color='b',
                marker='<',
                scatter_kws={'alpha': 1, 's':20})
    sns.regplot(x=np.array(df_tmp_0.indif_eta),
                y=np.array(df_tmp_0.min_max_val),
                fit_reg=False,
                y_jitter=0.05,
                ax=ax[c,1],
                label='Lower Bound',
                color='orange',
                marker='>',
                scatter_kws={'alpha': 1, 's':20})

    ax[c,1].axhline(y=0.5,color='grey',linestyle='--')

    ax[c,1].axvline(x=x_test[idx_m], color='red', linestyle='--', label='Prediction')
    ax[c,1].axvline(x=condition, color='grey', linestyle='--', label='Growth optimal')
    ax[c,1].legend(loc='upper left')

    ax[c,1].set(title=f'Logistic regression',
        ylabel = '',
        xlabel = 'Indifference eta',
        yticks = [0, 0.5, 1],
        ylim = (-0.25, 1.25),
        xticks = np.linspace(-5,5,11),
        xlim = [-5,5])

    if hlm_samples_found and i > 3:
        #ax[c,2].axvline(x=x_test[idx_m], color='red', linestyle='--')
        mu_eta = HLM_samples['mu_eta'][:,:,c]
        sns.kdeplot(mu_eta.flatten(),ax=ax[c,2])
        xs, ys = ax[c,2].lines[-1].get_data()
        ax[c,2].fill_between(xs, ys, color='red', alpha=0.05)
        mode_idx = np.argmax(ys)
        ax[c,2].vlines(xs[mode_idx], 0, ys[mode_idx], ls='--', color='red', label='Prediction')
    ax[c,2].set(title=f'Bayesian Model',
                ylabel = '',
                xticks = np.linspace(-5,5,11),
                xlim = [-5,5],
                xlabel='Risk aversion estimate')
    ax[c,2].axvline(condition, linestyle='--', linewidth=1, color='grey', label ='Growth optimal')
    ax[c,2].legend(loc='upper left')


fig.tight_layout()
fig.savefig(os.path.join(save_path, f'active_results_aggregated.png'))