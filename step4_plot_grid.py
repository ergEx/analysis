from codebase.create_grid_sim_plot import create_grid_sim_plot
import traceback
from codebase.utils import get_config_filename

if __name__ == '__main__':
    import sys
    from codebase import write_provenance

    command = '\t'.join(sys.argv)
    print(sys.argv)
    write_provenance(command)
    try:
        config_file = get_config_filename(sys.argv)
        create_grid_sim_plot(config_file=config_file)
        write_provenance('executed successfully')
    except Exception as e:
        print(e)
        traceback.print_exc()
        write_provenance('FAILED!!')
