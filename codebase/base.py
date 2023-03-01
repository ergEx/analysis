import os


def get_config_filename(argv):
    # Determine the name of the config file to be used
    filename = "config_default.yaml"
    if len(argv) == 1:
        print("No config file specified. Assuming config_default.yaml")
    else:
        filename = argv[1]
        print("Using config file ", filename)

    # Check that the config file exists and is readable
    if not os.access(filename, os.R_OK):
        print("Config file ", filename, " does not exist or is not readable. Exiting.")
        exit()

    return filename
