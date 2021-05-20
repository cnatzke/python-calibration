###################################################
# Library to read in histogram data or raw data and create complicated histograms
###################################################
import numpy as np


class histogram_manager:

    def __init__(self, data_df, output_filename=None, energy_max=3000):
        self.data_df = data_df
        self.output_filename = output_filename
        self.energy_max = energy_max

    def generate_channel_histograms_1D(self, column):
        # read in dataframe with raw data (e.g. charge, time, channel, etc)
        # and create histograms for each channel
        energy_bins = np.linspace(
            0, self.energy_max, num=self.energy_max, dtype=int)
        channel_histogram = None
        channel_histograms_dict = {}
        channel_min = self.data_df.crystal.min()
        channel_max = self.data_df.crystal.max()

        # iterate over each crystal
        print(f'Building channel dependent {column} histograms ... ')
        for channel in np.arange(channel_min, channel_max + 1, 1):
            # extract crystal specific data
            channel_df = self.data_df[self.data_df['crystal'] == channel]
            if not channel_df.empty:
                channel_histogram = np.histogram(
                    channel_df[column], bins=energy_bins)[0]
                # append 0 to end of histogram to get bins to match up
                channel_histogram = np.append(channel_histogram, [0])
                channel_histograms_dict[channel] = channel_histogram
            else:
                print(f'  Channel {channel} is empty, skipping ...')
                continue
        return channel_histograms_dict
