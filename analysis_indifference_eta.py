import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api
from scipy.special import expit, logit
from statsmodels.tools import add_constant

root_path = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(root_path,'data','all_data.csv'), sep=',')

###create indifference eta and wealth plots
fig_dynamic, ax_dynamic = plt.subplots(1,len(set(df.eta)),figsize = (12,5))
fig_subject, ax_subject = plt.subplots(len(set(df.subject_id)),len(set(df.eta)),figsize = (12,5))
fig_subject_wealth, ax_subject_wealth = plt.subplots(len(set(df.subject_id)),len(set(df.eta)),figsize = (12,5))

for dynamic_idx, dynamic in enumerate(set(df.eta)):
    trial_counter = 0
    tick_place = [0]
    tick_names = []
    for subject_idx, subject in enumerate(set(df.subject_id)):
        trial_counter_participant = 0
        tick_names.append(f'Subject {subject}')
        df_tmp = df.query('eta == @dynamic and subject_id == @subject').reset_index(drop=True)
        for ii, choice in enumerate(df_tmp['selected_side_map']):
            trial_counter += 1
            trial_counter_participant += 1
            trial = df_tmp.loc[ii,:]
            if np.isnan(trial.indif_eta):
                continue

            ax_subject[subject_idx,dynamic_idx].plot(trial.indif_eta, ii, marker=trial.min_max_sign, color = trial.min_max_color)
            ax_dynamic[dynamic_idx].plot(trial.indif_eta, trial_counter, marker=trial.min_max_sign, color = trial.min_max_color)

            ax_subject[subject_idx,dynamic_idx].set(title = f'Subject {subject}, Dynamic {dynamic}', xlabel = 'Riskaversion ($\eta$)')
            ax_subject[subject_idx,dynamic_idx].axes.yaxis.set_visible(False)
            ax_subject[subject_idx,dynamic_idx].axvline(dynamic, linestyle='--', linewidth=1, color='k')

        ax_subject_wealth[subject_idx,dynamic_idx].plot(df_tmp.index,df_tmp.wealth)
        ax_subject_wealth[subject_idx,dynamic_idx].set(title = f'Subject {subject} wealth, Dynamic {dynamic}', xlabel = 'Trial', ylabel = 'w')

        ax_dynamic[dynamic_idx].axhline(trial_counter, linestyle='--', linewidth=1, color='k')
        tick_place.append(trial_counter - trial_counter_participant / 2)
    if dynamic_idx == 0:
        ax_dynamic[dynamic_idx].set_yticks(tick_place[1:])
        ax_dynamic[dynamic_idx].set_yticklabels(tick_names, minor=False)
    else:
        ax_dynamic[dynamic_idx].axes.yaxis.set_visible(False)

    ax_dynamic[dynamic_idx].set(title = dynamic, xlabel = 'Riskaversion ($\eta$)')
    ax_dynamic[dynamic_idx].axvline(dynamic, linestyle='--', linewidth=1, color='k')
plt.tight_layout()
plt.show()

df = df.query('indif_eta.notnull()', engine='python')
#Logistic regression
fig_subject, ax_subject = plt.subplots(len(set(df.subject_id)),len(set(df.eta)),figsize = (12,5))
df_output = pd.DataFrame(columns=('Subject','Risk_aversion_estimate_add','Risk_aversion_estimate_mul'))

for subject_idx, subject in enumerate(set(df.subject_id)):
    risk_aversion_estimate = dict()
    for dynamic_idx, dynamic in enumerate(set(df.eta)):
        df_tmp = df.query('eta == @dynamic and subject_id == @subject').reset_index(drop=True)
        df_tmp = df_tmp.sort_values(by=['indif_eta'])
        x = np.array(df_tmp.indif_eta)
        X = add_constant(x)
        x_test = np.linspace(min(x), max(x), len(x)*5)
        X_test = add_constant(x_test)
        y = np.array(df_tmp.min_max_val)
        model = statsmodels.api.Logit(y, X).fit(disp=0)
        pred = model.predict(X_test)
        se = np.sqrt(np.array([xx@model.cov_params()@xx for xx in X_test]))

        tmp = {'x':x_test, 'pred':pred, 'ymin': expit(logit(pred) - 1.96*se), 'ymax': expit(logit(pred) + 1.96*se)}

        ax_subject[subject_idx,dynamic_idx].fill_between(x_test, tmp['ymin'], tmp['ymax'], where=tmp['ymax'] >= tmp['ymin'],
                 facecolor='grey', interpolate=True, alpha=0.5,label='95 % confidence interval')

        ax_subject[subject_idx,dynamic_idx].plot(tmp['x'],tmp['pred'],color='black')

        sns.regplot(x=x,
                    y=y,
                    fit_reg=False,
                    y_jitter=0.05,
                    ax=ax_subject[subject_idx,dynamic_idx],
                    label='data',
                    color='grey',
                    scatter_kws={'alpha': 1, 's':20})

        ax_subject[subject_idx,dynamic_idx].legend(
                    loc='lower right',
                    fontsize='small',
                )
        ax_subject[subject_idx,dynamic_idx].axhline(y=0.5,color='grey',linestyle='--')

        ax_subject[subject_idx,dynamic_idx].set(title=f'Subject {subject}, dynamic = {dynamic}',
                                                ylabel = 'y',
                                                xlabel = 'Indifference eta',
                                                yticks = [0, 0.5, 1],
                                                ylim = (-0.25, 1.25))
        risk_aversion_estimate[dynamic] = -model.params[0]/model.params[1]
    new_row = {'Subject':subject,'Risk_aversion_estimate_add':risk_aversion_estimate[0.0],'Risk_aversion_estimate_mul':risk_aversion_estimate[1.0]}
    df_output = df_output.append(new_row, ignore_index=True)
plt.tight_layout()
plt.show()

df_output.to_csv(os.path.join(root_path, 'data', 'logistic_regression_output.csv'), sep='\t')
df_output.to_csv(os.path.join(root_path, 'data', 'backup', f'logistic_regression_output{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'), sep='\t')
