# *************************************************************************************
# * Written by : Connor Natzke
# * Started : May 2021 - Still during the plague..
# * Purpose : Check how good the GRIFFIN energy calibration is
#  * Requirements : Python 3, matplotlib, probably something other stuff numpy,scipy...
# *************************************************************************************
from lib.input_handler import input_handler
from lib.energy_calibration import energy_calibration

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    input_file = './data/data_60Co.csv'
    cal_file = './linear_calibration.cal'

    input_man = input_handler()
    data_df = input_man.read_in_data(input_file)

    energy_cal = energy_calibration(cal_file=cal_file)
    energy_cal.apply_calibration(data_df)

    # --- Plot testing -----------------------------------------
    print("Plotting calibrated energies ... ")
    energy_bins = np.linspace(0, 3000, 3000, dtype=int)
    channel_bins = np.linspace(0, 67, 67, dtype=int)
    sns.set_style('ticks')
    sns.set_context('notebook')
    width, height = plt.figaspect(0.563)  # 16x9
    fig, axes = plt.subplots(num=None, figsize=(width, height), dpi=96)

    plt.hist2d(data_df.crystal, data_df.energy, bins=[channel_bins, energy_bins], cmap="viridis", norm=LogNorm())
    plt.colorbar(use_gridspec=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
