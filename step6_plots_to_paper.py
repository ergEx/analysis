import os
import shutil



def copy_file(path_in, file_in, path_out, file_out, dryrun=False):

    if not os.path.isfile(os.path.join(path_in, file_in)):
        raise FileNotFoundError(f'File {file_in} does not exist!')

    if not os.path.isdir(path_out):
        raise FileNotFoundError('Out folder does not exist!')

    file_source = os.path.join(path_in, file_in)
    file_target = os.path.join(path_out, file_out)

    if not dryrun:
        shutil.copyfile(src=file_source, dst=file_target)



def main(dryrun):
    path_out = 'paper_figures'

    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    path_support = 'figs/support'
    path_data = 'figs/2_full_data/'
    path_pilot = 'figs/1_pilot/'
    path_cph = 'figs/3_CPH/'
    path_sims = 'figs/0_simulation/grid'
    path_r = 'r_analyses/'

    match_data = {
        '01_passive_trajectories.pdf': 'fig2_passive_trajectories.pdf',
        '02_active_trajectories.pdf': 'fig5_active_trajectories.pdf',
        '04_riskaversion_bayesian_1.pdf': 'fig7_riskaversion_nofull_1d.pdf',
        '04_riskaversion_bayesian_2.pdf': 'fig7_riskaversion_nofull_2d.pdf',
        '04_riskaversion_bayesian_3.pdf': 'fig7_riskaversion_partial_1d.pdf',
        '04_riskaversion_bayesian_4.pdf': 'fig7_riskaversion_partial_2d.pdf',
        '08_q2_pairwise_diff_partial_pooling.pdf' : 'fig8_pairwise_add_mult.pdf',
        '07_model_selection_1_1.pdf' : 'fig8_selection_weakee_eut.pdf',
        '08_q1_pairwise_diff_partial_pooling.pdf' : 'fig9_pairwise_ee_eut.pdf',
        '07_model_selection_0_1.pdf' : 'fig9_selection_ee_eut.pdf',
        '08_q3_corelation_partial_pooling.pdf' : 'fig10_correlation.pdf',
        '06_sensitivity_bayesian_1.pdf': 'figS6_sensitivity1.pdf',
        '06_sensitivity_bayesian_2.pdf': 'figS6_sensitivity2.pdf',
        '05_riskaversion_mcmc_samples_1.pdf': 'figS5_mcmc1.pdf',
        '05_riskaversion_mcmc_samples_2.pdf': 'figS5_mcmc2.pdf',
        '08_model_selection_data_pooling.pdf': 'figS7_data_pooling.pdf'

        }

    match_pilot = {
        '04_riskaversion_bayesian_3.pdf': 'fig11_pilot_riskaversion_partial_1d.pdf',
        '04_riskaversion_bayesian_4.pdf': 'fig11_pilot_riskaversion_partial_2d.pdf',
        '08_q2_pairwise_diff_partial_pooling.pdf' : 'fig11_pilot_pairwise_add_mult.pdf',
        '07_model_selection_0_1.pdf' : 'fig11_pilot_selection_weakee_eut.pdf',

    }

    match_cph = {
        '04_riskaversion_bayesian_3.pdf': 'fig11_cph_riskaversion_partial_1d.pdf',
        '04_riskaversion_bayesian_4.pdf': 'fig11_cph_riskaversion_partial_2d.pdf',
        '08_q2_pairwise_diff_partial_pooling.pdf' : 'fig11_cph_pairwise_add_mult.pdf',
        '07_model_selection_0_1.pdf' : 'fig11_cph_selection_weakee_eut.pdf',

    }
    match_support = {
        'EE_pred.pdf' : 'fig6_ee.pdf',
        'EE2_pred.pdf' : 'fig6_weakee.pdf',
        'EUT_pred.pdf' : 'fig6_eut.pdf',
        'bracketing_1.pdf': 'figS3_bracketing_1.pdf',
        'bracketing_2.pdf': 'figS3_bracketing_2.pdf',
        'bracketing_3.pdf': 'figS3_bracketing_3.pdf',
        'utility_figure-0.5.png': 'figS2_utility_1.png',
        'utility_figure0.5.png': 'figS2_utility_2.png',}

    match_r = {
        'plot_bfda_h0.pdf': 'figS1_bfdah0.pdf',
        'plot_bfda_h1.pdf': 'figS1_bfdah1.pdf'
    }

    match_sims = {
        'simulations_bayesian_partial_pooling.pdf': 'figS4_partial_pooling.pdf',
        'simulations_bayesian_no_pooling.pdf': 'figS4_no_pooling.pdf',
        'simulations_bracketing.pdf': 'figS4_bracketing.pdf'
    }
    dict_list = [match_data, match_support, match_cph, match_pilot, match_r, match_sims]
    path_list = [path_data, path_support, path_cph, path_pilot, path_r, path_sims]


    for dl, pl in zip(dict_list, path_list):

        for sf, tf in dl.items():
            copy_file(pl, sf, path_out, tf, dryrun=dryrun)

if __name__ == '__main__':
    import sys
    from codebase.utils import write_provenance

    command = '\t'.join(sys.argv)
    print(sys.argv)

    write_provenance(command)
    if len(sys.argv) == 1:
        dr = 0
    else:
        dr = sys.argv[1]

    try:
        main(dr)
        write_provenance('executed successfully')
    except Exception as e:
        print(e)
        write_provenance('FAILED!!')




