import pandas as pd
import numpy as np


data_path = ''

sub = pd.read_csv(data_path + 'participants.tsv', sep='\t')
sub = sub.loc[sub.exclusion.isin([0, 3]), :]

all_data  = pd.read_csv('../data/2_full_data/all_active_phase_data_w_indif_etas.csv', sep='\t')

# all_data['luck'] = all_data['chosen_expected_gamma'] < all_data['realized_gamma']
all_data['luck'] =  all_data['realized_gamma'] - all_data['chosen_expected_gamma']
all_data['wealth'] = all_data['wealth'] + all_data['delta_wealth']
all_data['log_wealth'] = np.log(all_data['wealth'])

luck = all_data.groupby(['participant_id', 'eta']).sum().reset_index()[['participant_id', 'eta', 'luck']]
wealth_var = all_data.groupby(['participant_id', 'eta']).var().reset_index()[['participant_id', 'eta', 'wealth']]
log_wealth_var = all_data[np.isfinite(all_data.log_wealth)].groupby(['participant_id', 'eta']).var().reset_index()

final_wealth = all_data.query('trial==159')[['wealth', 'participant_id', 'eta']]

jasp_input = pd.read_csv('../data/2_full_data/jasp_input.csv', sep='\t')
jasp_input.participant_id = jasp_input.participant_id.str.lower()
jasp_input['var_wealth_eta0.0'] = wealth_var.query('eta == 0.0').wealth.values
jasp_input['var_wealth_eta1.0'] = wealth_var.query('eta == 1.0').wealth.values
jasp_input['log_var_wealth_eta0.0'] = log_wealth_var.query('eta == 0.0').log_wealth.values
jasp_input['log_var_wealth_eta1.0'] = log_wealth_var.query('eta == 1.0').log_wealth.values
jasp_input['final_wealth_eta0.0'] = final_wealth.query('eta == 0.0').wealth.values
jasp_input['final_wealth_eta1.0'] = final_wealth.query('eta == 1.0').wealth.values
jasp_input['luck_eta0.0'] = luck.query('eta == 0.0').luck.values
jasp_input['luck_eta1.0'] = luck.query('eta == 1.0').luck.values

jasp_input['d_luck'] = jasp_input['luck_eta1.0'] - jasp_input['luck_eta0.0']
jasp_input['d_final_wealth'] = jasp_input['final_wealth_eta1.0'] - jasp_input['final_wealth_eta0.0']
jasp_input['d_eta'] = jasp_input['1.0_partial_pooling'] - jasp_input['0.0_partial_pooling']
jasp_input['d_log_var_wealth'] = jasp_input['log_var_wealth_eta1.0'] - jasp_input['log_var_wealth_eta0.0']
jasp_input['d_var_wealth'] = jasp_input['var_wealth_eta1.0'] - jasp_input['var_wealth_eta0.0']

sub_scales = ['E', 'F,G', 'F,I', 'H,S', 'R', 'S']
sub_scales = [ii.replace(',', '') for ii in sub_scales]
output_df = {ii : [] for ii in ['sub', 'general_risk'] + sub_scales}
output_df_cols = list(output_df.keys())

summed_responses = []

for ii in sub.participant_id:
    preference = pd.read_csv( data_path + f'sub-{ii}/ses-2/sub-{ii}_dospert-risk-taking.tsv', sep='\t')
    benefits = pd.read_csv( data_path + f'sub-{ii}/ses-2/sub-{ii}_dospert-risk-benefits.tsv', sep='\t')
    perception = pd.read_csv( data_path + f'sub-{ii}/ses-2/sub-{ii}_dospert-perceived-risk.tsv', sep='\t')

    risk_prop = pd.read_csv( data_path + f'sub-{ii}/ses-2/sub-{ii}_risk-propensity.tsv', sep='\t')
    risk_prop['score'] = risk_prop['response']
    risk_prop.loc[[0, 1, 2, 4], 'score']  = 10 - risk_prop.loc[[0, 1, 2, 4], 'response']
    risk_prop['score'].sum()

    full_dospert = preference[['coding', 'response']].copy()
    full_dospert = full_dospert.rename(columns={'response': 'preference'})
    full_dospert['coding'] = full_dospert['coding'].str.replace(',', '')
    full_dospert['benefits'] = benefits['response']
    full_dospert['perception'] = perception['response']
    full_dospert['intercept'] = 1
    summed_response = full_dospert.groupby('coding').sum().reset_index()
    summed_response = summed_response.melt(id_vars='coding', value_vars=['perception', 'preference', 'benefits'], value_name='score')
    summed_response['coding'] = summed_response.variable + '_'  + summed_response.coding
    summed_response = summed_response[['coding', 'score']].T
    summed_response = summed_response.rename(columns=summed_response.iloc[0]).drop(summed_response.index[0])
    summed_response[['total_preference', 'total_perception', 'total_benefits']] = full_dospert[['preference', 'perception', 'benefits']].sum().T
    summed_response['risk_propensity'] = risk_prop.score.sum()
    summed_response['participant_id'] = ii

    summed_responses.append(summed_response)


dospert_scores = pd.concat(summed_responses, axis=0, ignore_index=True)
jasp_input = jasp_input.merge(dospert_scores, on='participant_id')


anova = jasp_input.copy()
anova = anova[['d_luck', 'd_final_wealth', 'd_eta', 'd_log_var_wealth', 'd_var_wealth']]

jasp_input = jasp_input.merge(sub[['participant_id', 'Age', 'Sex', 'Income']], on='participant_id')

incl = ['.0_partial', 'preference_', 'perception_', 'benefits_', 'propensity', 'Age', 'Sex', 'Income']

incl_cols = [i for i in jasp_input.columns if any([j in i for j in incl])]

print(incl_cols)
jasp_input[incl_cols].to_csv('full_data_regression.tsv', sep='\t', index=False)

anova.to_csv('full_data_anova.tsv', sep='\t', index=False)