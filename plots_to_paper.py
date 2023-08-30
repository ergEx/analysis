import os
import shutil



def copy_file(path_in, file_in, path_out, file_out, dryrun=False):

    if not os.path.isfile(os.path.join(path_in, file_in)):
        raise FileNotFoundError('File does not exist!')

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
    path_pilot = 'figs/1_pilot'
    path_sim_grid = 'figs/0_simulation/grid'
    path_r_analysis = 'r_analyses'

    match_support = {
        'Fig2_a_h1.pdf' : 'h_1.pdf',
        'Fig2_b_h2.pdf' : 'h_2.pdf',
        'Fig2_c_h0.pdf' : 'h_0.pdf',
        'FigC1_a_eut_fig.png': 'utility_figure.png',
        'FigC1_b_eut_eta_m1.png': 'utility_figure_eta_-1.0.png',
        'FigC1_c_eut_eta_1.png': 'utility_figure_eta_1.0.png',
        'FigD1_a_bracketing1.pdf' : 'bracketing_1.pdf',
        'FigD1_b_bracketing2.pdf' : 'bracketing_2.pdf',
        'FigD1_c_bracketing3.pdf' : 'bracketing_3.pdf',
        }

    match_pilot = {
        'Fig3_a_riskaversion_partial_pooling_group_bayesian.pdf'  : '05a_riskaversion_partial_pooling_group_bayesian.pdf',
        'Fig3_b_riskaversion_partial_pooling_individual_bayesian.pdf' : '05b_riskaversion_partial_pooling_individual_bayesian.pdf',
        'Fig3_c_raincloud_distance_partial_pooling.pdf' : '05f_raincloud_distance_partial_pooling.pdf',
        'Fig3_d_q1_sequential_partial_pooling.pdf' : 'q1_sequential_partial_pooling.pdf',
        'Fig3_e_q2_sequential_partial_pooling.pdf' : 'q2_sequential_partial_pooling.pdf',
        'Fig3_f_correlation_riskaversion_partial_pooling.pdf' : '05d_correlation_riskaversion_partial_pooling.pdf',
        'FigF1_a_riskaversion_full_pooling_group_bayesian.pdf' : '04a_riskaversion_full_pooling_group_bayesian.pdf',
        'FigF1_b_riskaversion_full_pooling_group_bracketing.pdf' : '02a_riskaversion_full_pooling_group_bracketing.pdf',
        'FigF1_c_riskaversion_no_pooling_individual_bayesian.pdf' : '06b_riskaversion_no_pooling_individual_bayesian.pdf',
        'FigF1_d_riskaversion_no_pooling_individual_bracketing.pdf' : '03b_riskaversion_no_pooling_individual_bracketing.pdf'
    }

    match_simulation_grid = {
        'FigC2_a_grid_simulation_riskaversion_partial_pooling.pdf' : 'grid_simulation_riskaversion_partial_pooling.pdf',
        'FigC2_b_grid_simulation_riskaversion_no_pooling.pdf' : 'grid_simulation_riskaversion_no_pooling.pdf',
        'FigC2_c_grid_simulation_riskaversion_bracketing.pdf' : 'grid_simulation_riskaversion_bracketing.pdf'
    }

    match_r_analyses = {
        'FigE1_a_bfdah0.pdf' : 'plot_bfda_h0.pdf',
        'FigE1_b_bfdah1.pdf' : 'plot_bfda_h1.pdf',
    }


    dict_list = [match_support, match_simulation_grid, match_r_analyses, match_pilot]
    path_list = [path_support, path_sim_grid, path_r_analysis, path_pilot]


    for dl, pl in zip(dict_list, path_list):

        for tf, sf in dl.items():
            copy_file(pl, sf, path_out, tf, dryrun=dryrun)

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        dr = 0
    else:
        dr = sys.argv[1]

    main(dr)




