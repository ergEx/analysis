import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api
from mpl_toolkits import mplot3d
from scipy.special import expit, logit
from sklearn.metrics import confusion_matrix
from statsmodels.tools import add_constant

RESET = 45
root_path = os.path.dirname(__file__)
design_variant = 'test'

condition_specs = {0.0:'Additive', 1.0:'Multiplicative'}

passive_phase_df = pd.read_csv(os.path.join(root_path,'data','experiment_output',design_variant,'all_passive_phase_data.csv'), sep='\t')
active_phase_df = pd.read_csv(os.path.join(root_path,'data','experiment_output',design_variant,'all_active_phase_data.csv'), sep='\t')

logistic_regression_outout = pd.DataFrame(columns=['Subject',
                                                   'Risk_aversion_estimate_Additive_low_conf',
                                                   'Risk_aversion_estimate_Additive',
                                                   'Risk_aversion_estimate_Additive_upper_conf',
                                                   'Risk_aversion_estimate_Multiplicative_low_conf',
                                                   'Risk_aversion_estimate_Multiplicative',
                                                   'Risk_aversion_estimate_Multiplicative_upper_conf'])

for c,condition in enumerate(set(passive_phase_df['eta'])):
    for i,subject in enumerate(set(passive_phase_df['participant_id'])):
        save_path = os.path.join(root_path, 'figs', design_variant, str(subject))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        '''PASIVE PHASE'''
        passive_subject_df = passive_phase_df.query('participant_id == @subject and eta == @condition').reset_index(drop=True)
        fig, ax = plt.subplots(1,1)
        ax.plot(passive_subject_df.trial, passive_subject_df.wealth)
        ax.set(title=f'Passive phase \nSubject {subject}, Condition: {condition_specs[condition]}',
               xlabel='Trial',
               ylabel='Wealth')
        for reset_idx in range(1,4):
            ax.axvline(x=RESET*reset_idx,color='grey',linestyle='--')
        ax.plot([], label="Reset",color='grey',linestyle='--')
        ax.legend()

        fig.savefig(os.path.join(save_path, f'Passive_trajectory_{condition_specs[condition]}.png'))
        plt.close(fig)

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

        #Indifference eta plots
        fig, ax = plt.subplots(1,1)
        for ii, choice in enumerate(active_subject_df['selected_side_map']):
            trial = active_subject_df.loc[ii,:]
            if np.isnan(trial.indif_eta):
                continue

            ax.axvline(condition, linestyle='--', linewidth=1, color='k')

            ax.plot(trial.indif_eta, ii, marker=trial.min_max_sign, color=trial.min_max_color)

            ax.set(title = f'Indifference eta \nSubject {subject}, Condition: {condition_specs[condition]}',
                   xlabel = 'Riskaversion ($\eta$)')

            ax.axes.yaxis.set_visible(False)

        fig.savefig(os.path.join(save_path, f'Indifference_eta_{condition_specs[condition]}.png'))
        ax.set_xlim([-2,3])
        fig.savefig(os.path.join(save_path, f'Indifference_eta_zoom_{condition_specs[condition]}.png'))
        plt.close(fig)

        #Indifference eta logistic regression
        fig, ax = plt.subplots(1,1)
        df_tmp = active_subject_df.query('indif_eta.notnull()', engine='python')
        model = statsmodels.api.Logit(np.array(df_tmp.min_max_val), add_constant(np.array(df_tmp.indif_eta))).fit(disp=0)
        x_test = np.linspace(min(df_tmp.indif_eta), max(df_tmp.indif_eta), len(df_tmp.indif_eta)*5)
        X_test = add_constant(x_test)
        pred = model.predict(X_test)
        se = np.sqrt(np.array([xx@model.cov_params()@xx for xx in X_test]))

        tmp = {'x':x_test, 'pred':pred, 'ymin': expit(logit(pred) - 1.96*se), 'ymax': expit(logit(pred) + 1.96*se)}

        ax.fill_between(x_test, tmp['ymin'], tmp['ymax'], where=tmp['ymax'] >= tmp['ymin'],
                    facecolor='grey', interpolate=True, alpha=0.5,label='95 % confidence interval')

        ax.plot(tmp['x'],tmp['pred'],color='black')

        sns.regplot(x=np.array(df_tmp.indif_eta),
                    y=np.array(df_tmp.min_max_val),
                    fit_reg=False,
                    y_jitter=0.05,
                    ax=ax,
                    label='data',
                    color='grey',
                    scatter_kws={'alpha': 1, 's':20})

        ax.axhline(y=0.5,color='grey',linestyle='--')

        ax.set(title=f'Subject {subject}, dynamic = {condition_specs[condition]}',
               ylabel = 'y',
               xlabel = 'Indifference eta',
               yticks = [0, 0.5, 1],
               ylim = (-0.25, 1.25))

        fig.savefig(os.path.join(save_path, f'Indifference_eta_logistic_regression_{condition_specs[condition]}.png'))
        ax.set_xlim([-3,4])
        fig.savefig(os.path.join(save_path, f'Indifference_eta_logistic_regression_zoom_{condition_specs[condition]}.png'))
        plt.close(fig)

        idx_m = min([i for i in range(len(tmp['pred'])) if tmp['pred'][i] > 0.5]) if len([i for i in range(len(tmp['pred'])) if tmp['pred'][i] > 0.5]) > 0 else len(tmp['x']) -1
        idx_l = min([i for i in range(len(tmp['ymin'])) if tmp['ymin'][i] > 0.5]) if len([i for i in range(len(tmp['ymin'])) if tmp['ymin'][i] > 0.5]) > 0 else len(tmp['x']) -1
        idx_h = min([i for i in range(len(tmp['ymax'])) if tmp['ymax'][i] > 0.5]) if len([i for i in range(len(tmp['ymax'])) if tmp['ymax'][i] > 0.5]) > 0 else len(tmp['x']) -1

        if c == 0:
            logistic_regression_outout.loc[i] = pd.Series(dtype=float)
        logistic_regression_outout.at[i,'Subject'] = subject
        logistic_regression_outout.at[i,f'Risk_aversion_estimate_{condition_specs[condition]}_low_conf'] = tmp['x'][idx_h]
        logistic_regression_outout.at[i,f'Risk_aversion_estimate_{condition_specs[condition]}'] = tmp['x'][idx_m]
        logistic_regression_outout.at[i,f'Risk_aversion_estimate_{condition_specs[condition]}_upper_conf'] = tmp['x'][idx_l]

        #Two variable logistic regression
        model =  statsmodels.api.Logit(active_subject_df[['choose_high_variance']], active_subject_df[['dEV', 'dvar' ]]).fit()

        pred = model.predict(active_subject_df[['dEV', 'dvar' ]])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xline = np.linspace(min(active_subject_df.dEV), max(active_subject_df.dEV), len(active_subject_df.dEV))
        yline = np.linspace(min(active_subject_df.dvar),max(active_subject_df.dvar),len(active_subject_df.dvar))
        zline = pred
        ax.plot3D(xline, yline, zline, 'gray')
        ax.set(title = f'Subject {subject}, dynamic = {condition_specs[condition]}',
               xlabel = '$\Delta$EV',
               ylabel = '$\Delta$var',
               zlabel = 'Choice probability',
               zlim = [0,1])
        fig.savefig(os.path.join(save_path, f'two_variable_logistic_regression_{condition_specs[condition]}.png'))

        #Signal detection
        df_tmp = active_subject_df.query('response_time_optimal.notnull() and response_time_optimal != 0', engine='python')
        conf_matrix = confusion_matrix(df_tmp.response_time_optimal.map({-1:0,1:1}), df_tmp.selected_side_map)



logistic_regression_outout.to_csv(os.path.join(root_path,'data','experiment_output',design_variant, 'logistic_regression_output.csv'), sep='\t')
