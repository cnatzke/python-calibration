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


from pprint import pprint
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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

    '''
    energy_cal_1 = energy_calibration(cal_file='./my_cal.cal')
    # need to fix calibration
    energy_cal_1.apply_calibration(mydata_df)

    # --- Plot testing -----------------------------------------
    energy_bins = np.linspace(0, 3000, 3000, dtype=int)
    channel_bins = np.linspace(0, 70, 70, dtype=int)
    sns.set_style('ticks')
    sns.set_context('notebook')
    width, height = plt.figaspect(0.563)  # 16x9
    fig, axes = plt.subplots(num=None, figsize=(width, height), dpi=96)

    plt.hist2d(mydata_df.crystal, mydata_df.energy, bins=[channel_bins, energy_bins], cmap="viridis", norm=LogNorm())
    plt.colorbar(use_gridspec=True)
    plt.tight_layout()
    plt.show()
    '''


def main():
    parser = argparse.ArgumentParser(description='GRIFFIN Energy Calibrator')

    parser.add_argument('--source_file', '-s', dest='config_file', required=True, help="Name of source file")
    parser.add_argument('--cal_output_file', dest='cal_output_file', required=False, help="Name of output calibration file")

    args, unknown = parser.parse_known_args()

    parse_and_run(args)

if __name__ == "__main__":
    main()
