import sys
import time
import yaml
from codebase import readingdata, utils

def main():
    config_file = utils.get_config_filename(sys.argv)

    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_type = config["data_type"]
    data_variant = config["data_variant"]

    start_time = time.time()
    print(f"\n--- {time.ctime(start_time)} --- ")
    print(f"\nReading data")
    print(f"Data: {data_type} \nVariant: {data_variant}")

    readingdata.main(config_file)

    print(f"\n--- Code ran in {(time.time() - start_time):.2f} seconds ---")

    #config['readingdata']['run'] = False

    # Not sure about this ;)
    #with open(f"{config_file}", 'w') as outfile:
    #    yaml.dump(config, outfile, default_flow_style=False)


if __name__ == "__main__":
    import sys
    from codebase.utils import write_provenance

    command = '\t'.join(sys.argv)
    print(sys.argv)
    write_provenance(command)
    try:
        main()
        write_provenance('executed successfully')
    except:
        write_provenance('FAILED!!')
