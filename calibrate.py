# *************************************************************************************
# * Written by : Connor Natzke
# * Started : May 2021 - Still during the plague..
# * Purpose : Calibrate GRIFFIN data using quadratic calibration
#  * Requirements : Python 3, matplotlib, probably something other stuff numpy,scipy...
# *************************************************************************************
import argparse
from configparser import ConfigParser
from pathlib import Path
from lib.energy_calibration import energy_calibration


def parse_and_run(args):

    config = ConfigParser()
    config.read(args.config_file)
    TOP_DIR = Path(__file__).resolve().parent

    # since we calibrate with more than one source we build a dict with the source name and file location
    source_dict = {}
    for key in config['Sources']:
        source_dict[key] = f"{config['Basics']['data_dir']}/{config['Sources'][key]}"

    energy_cal = energy_calibration()
    energy_cal.calibrate(source_dict)


def main():
    parser = argparse.ArgumentParser(description='GRIFFIN Energy Calibrator')

    parser.add_argument('--source_file', '-s', dest='config_file', required=True, help="Name of source file")
    parser.add_argument('--cal_output_file', dest='cal_output_file', required=False, help="Name of output calibration file")

    args, unknown = parser.parse_known_args()

    parse_and_run(args)


if __name__ == "__main__":
    main()
