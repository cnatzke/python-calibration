from lib.input_handler import input_handler
from lib.histogram_manager import histogram_manager
from lib.energy_calibration import energy_calibration
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    input_file = './data_60Co_charge.csv'
    my_input = input_handler(input_file)
    mydata_df = my_input.read_in_data()

    hist_man = histogram_manager(mydata_df)
    charge_hist_dict = hist_man.generate_channel_histograms_1D('charge')

    energy_cal = energy_calibration()
    #energy_cal.calibrate(charge_hist_dict, 'linear', '60Co')

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

    plt.hist2d(mydata_df.crystal, mydata_df.calibrated_energy, bins=[channel_bins, energy_bins], cmap="viridis", norm=LogNorm())
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
