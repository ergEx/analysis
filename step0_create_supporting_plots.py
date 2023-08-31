from codebase import runBayesAnalysis
import yaml
import os
import time
from codebase.support_figures import plot_EUT_figure, plot_bracketing_fig, plot_hypothesis_fig

def main():

    start_time = time.time()

    fig_dir = 'figs/support/'

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    plot_bracketing_fig(fig_dir)
    plot_EUT_figure(fig_dir)
    plot_hypothesis_fig(fig_dir)

    print(f"\n--- Code ran in {(time.time() - start_time):.2f} seconds ---")


if __name__ == "__main__":
    import sys
    from codebase.utils import write_provenance

    command = '\t'.join(sys.argv)    
    write_provenance(command)

    main()

    write_provenance('executed successfully')