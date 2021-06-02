###################################################
# Library to read in histogram data or raw data and create complicated histograms
###################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class histogram_manager:

    def __init__(self, data_df, output_filename=None, energy_max=3000):
        self.data_df = data_df
        self.output_filename = output_filename
        self.energy_max = energy_max

    def generate_channel_histograms_1D(self, column):
        # read in dataframe with raw data (e.g. charge, time, channel, etc)
        # and create histograms for each channel
        energy_bins = np.linspace(0, self.energy_max, num=self.energy_max, dtype=int)
        channel_min = self.data_df.crystal.min()
        channel_max = self.data_df.crystal.max()
        channel_histogram_list = [0] * (channel_max + 1)  # GRIFFIN channels indexed [1,64] instead of [0,63]

        # iterate over each crystal
        print(f'Building channel dependent {column} histograms ... ')
        for channel in np.arange(channel_min, channel_max + 1, 1):
            # extract crystal specific data
            channel_df = self.data_df[self.data_df['crystal'] == channel]
            if not channel_df.empty:
                # build histogram
                channel_histogram, channel_histogram_bins = np.histogram(channel_df[column], bins=energy_bins)
                # get std error in bins
                channel_histogram_error = np.sqrt(channel_histogram)
                # append 0 to end of histogram to get bins to match up
                channel_histogram = np.append(channel_histogram, [0])
                channel_histogram_error = np.append(channel_histogram_error, [0])
                channel_histogram_dict = {'channel': channel,
                                          'counts': channel_histogram,
                                          'bins': channel_histogram_bins,
                                          'error': channel_histogram_error}
                channel_histogram_list[channel] = channel_histogram_dict
            else:
                print(f'  Channel {channel} is empty, skipping ...')
                continue
        # removing zero entries
        channel_histogram_list = list(filter(lambda entry: entry != 0, channel_histogram_list))
        return channel_histogram_list

    def draw_histogram(self, counts, bins, filename, title=None, peak_labels=None):
        # draws and saves histogram

        # basic formatting
        sns.set_style('ticks')
        sns.set_context('notebook')
        width, height = plt.figaspect(0.563)  # 16x9
        fig, axes = plt.subplots(num=None, figsize=(width, height), dpi=96)

        # plot histogram
        axes.step(bins, counts, where='mid')
        # labels peaks
        if peak_labels is not None:
            y_max = np.max(counts)
            self.label_peaks(axes, peak_labels, y_max)

        plt.yscale('log')
        plt.xlabel('Charge')
        plt.ylabel('Counts per 1 keV')
        axes.set_xlim(10, 1500)
        plt.title(title)
        plt.tight_layout()
        fig.savefig(filename)
        plt.close()
        return

    def label_peaks(self, axes, peak_labels, y_max):
        # Draw a line at the centre of a peak and label its xvalue
        for my_peak_index in range(len(peak_labels)):
            my_peak_energy = peak_labels[my_peak_index]['peak_energy']
            my_peak_centroid = peak_labels[my_peak_index]['fit_peak_centroid']
            axes.axvline(x=int(my_peak_centroid) + 1, color='red', linestyle='--', ymin=0, ymax=.75, lw=1, alpha=0.5)
            axes.text(int(my_peak_centroid), 0.25 * y_max, str(my_peak_energy), rotation=45, alpha=0.5, color='red')
        return
