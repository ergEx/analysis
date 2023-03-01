import sys

from codebase import base


def main():
    config_file = base.get_config_filename(sys.argv)

    from codebase import readingdata

    readingdata.main(config_file)

    from codebase import create_plots

    create_plots.main(config_file)


if __name__ == "__main__":
    main()
