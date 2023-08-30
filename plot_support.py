from codebase import runBayesAnalysis
import yaml
import os
import time
from codebase.support_figures import plot_EUT_figure, plot_bracketing_fig, plot_hypothesis_fig

def main():

    start_time = time.time()
    config_file = 'config_files/config_1_pilot.yaml'

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    fig_dir = 'figs/support/'

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    runBayesAnalysis.main(config_file, fig_dir)
    plot_bracketing_fig(fig_dir)
    plot_EUT_figure(fig_dir)
    plot_hypothesis_fig(fig_dir)

    print(f"\n--- Code ran in {(time.time() - start_time):.2f} seconds ---")


if __name__ == "__main__":
    main()