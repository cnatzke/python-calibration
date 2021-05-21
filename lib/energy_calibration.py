import numpy as np
import scipy.signal
from configobj import ConfigObj
from lib.input_handler import input_handler
from lib.histogram_manager import histogram_manager
from lmfit import Model, Parameters
from scipy.special import erf

from pprint import pprint


class energy_calibration:

    def __init__(self, cal_file=None, max_energy=3000):
        self.cal_file = cal_file
        self.cal_dict = {}
        self.max_energy = max_energy
        self.linear_cal_list = []
        self.quad_cal_list = []
        self.quad_cal_data = []
        self.source_channel_list = []
        self.source_info = {}
        self.source_peak_finder_dict = {}

    def read_in_calibration_file(self):
        # read in calibration file and make a dict of coefficients
        print(f'Reading calibration file: {self.cal_file}')
        config = ConfigObj(self.cal_file)
        for channel in config.keys():
            self.cal_dict[int(config[channel]['channel'])] = np.array(config[channel]['cal_coeff'], dtype=float)
        return

    def calibrate(self, source_dict):
        # we use a two step calibration process:
        # 1 - Perform an initial linear calibration using 60Co
        # 2 - Perform quadratic calibration using variety of sources

        my_input = input_handler()

        # first perform linear calibration using 60Co
        data_df = my_input.read_in_data(source_dict['60co'])
        hist_man = histogram_manager(data_df)
        charge_hist_dict = hist_man.generate_channel_histograms_1D('charge')
        self.perform_linear_calibration(charge_hist_dict, '60co')
        # second perform quadratic calibration using other sources
        for source_key in source_dict.keys():
            if source_key == '60co':
                self.quad_cal_list.append({'source': source_key, 'cal_info': self.linear_cal_list})
                continue
            else:
                print(f"Extracting {source_key} peak positions ...")
                data_df = my_input.read_in_data(source_dict[source_key])
                hist_man = histogram_manager(data_df)
                charge_hist_dict = hist_man.generate_channel_histograms_1D('charge')
                self.get_quadratic_calibration_values(charge_hist_dict, source_key)
                self.quad_cal_list.append({'source': source_key, 'cal_info': self.source_channel_list})

        # now we perform the quadratic calibration
        self.perform_quadratic_calibration()
        return

    def perform_linear_calibration(self, histograms, source_name):
        # Creates initial 60Co linear calibration to make quadratic peak search easier
        index = 0
        print("Perfoming linear calibration ...")
        for channel_dict_key in histograms.keys():
            if sum(histograms[channel_dict_key]) > 10000:
                self.linear_cal_list.append({'channel': channel_dict_key, 'hist': histograms[channel_dict_key]})

                # find the peaks of interest
                self.find_peaks(index, self.linear_cal_list, source_name)
                self.find_peak_centroids(index, self.linear_cal_list)
                # now we fit a linear function for basic linear calibration
                self.find_poly_fit(index, self.linear_cal_list, 1)
                index = index + 1
            else:
                print(f"  Channel {channel_dict_key} rejected due to insufficient counts.")
        # write out calibration file
        self.write_calibration_file('linear_calibration.cal')
        return

    def perform_quadratic_calibration(self):
        # first we need to collect the peak data from all sources
        print("Perfoming linear calibration ...")
        self.combine_all_source_data()
        for channel_index in range(0, len(self.quad_cal_data)):
            poly_fit = np.polyfit(self.quad_cal_data[channel_index]['centroids'], self.quad_cal_data[channel_index]['energy'], 2)
            self.quad_cal_data[channel_index].update({'poly_fit': poly_fit})
        # write out calibration file
        self.write_calibration_file('quadratic_calibration.cal')
        return

    def combine_all_source_data(self):
        # first aggregate the channel data from all sources used
        self.quad_cal_data = [0] * 64
        for my_source_index in range(len(self.quad_cal_list)):
            my_source = self.quad_cal_list[my_source_index]['cal_info']
            index = 0
            for my_channel_index in range(len(my_source)):
                my_peak_info = my_source[my_channel_index]['peak_info']
                if self.quad_cal_data[index] == 0:
                    self.quad_cal_data[index] = {'channel': my_source[my_channel_index]['channel'], 'energy': [], 'centroids': []}
                # extract fitted peak centroids
                for my_peak_index in range(len(my_peak_info)):
                    self.quad_cal_data[index]['energy'].append(my_peak_info[my_peak_index]['peak_energy'])
                    self.quad_cal_data[index]['centroids'].append(my_peak_info[my_peak_index]['fit_peak_centroid'])
                index = index + 1
        # remove zero channels
        self.quad_cal_data = list(filter(lambda entry: entry != 0, self.quad_cal_data))
        return

    def find_poly_fit(self, index, hist_list, degree, overwrite=True):
        peak_info = hist_list[index]['peak_info']
        peak_centroids = []
        peak_energy = []
        for my_peak_index in range(len(peak_info)):
            peak_centroids.append(peak_info[my_peak_index]['fit_peak_centroid'])
            peak_energy.append(peak_info[my_peak_index]['peak_energy'])

        # least squares polynomial fit
        poly_fit = np.polyfit(peak_centroids, peak_energy, degree)
        hist_list[index].update({'poly_fit': poly_fit})
        return

    def find_peaks(self, index, histogram_list, source_name):
        histogram_dict = histogram_list[index]
        self.get_peak_info(source_name)
        tallest_peak = np.amax(histogram_dict['hist'])
        tallest_peak_index = np.where(histogram_dict['hist'] == tallest_peak)[0][0]
        prominence = tallest_peak * self.source_peak_finder_dict['prominence_fraction_of_tallest_peak']
        peak_indices, peak_properties = scipy.signal.find_peaks(histogram_dict['hist'], prominence=prominence)

        # checking to make sure our found peaks make sense
        if len(peak_indices) < self.source_peak_finder_dict['num_peaks_needed']:
            print(f"Could not find {self.source_peak_finder_dict['num_peaks_needed']}, found {len(peak_indices)}")
            exit(1)
        try:
            tallest_peak_index_in_found_peaks = np.where(peak_indices == tallest_peak_index)[0][0]
        except IndexError:
            print(f'Tallest peak index: {tallest_peak_index}')
            print('Could not match tallest peak to a found peak ... something has gone horribly wrong')
            exit(1)

        peak_info = []
        for my_peak_index in range(0, len(peak_indices)):
            # print("My peak index", my_peak_index)
            peak_info.append({'peak_energy': self.source_info['peak'][my_peak_index],
                              'est_peak_centroid': peak_indices[my_peak_index],
                              'fit_peak_centroid': None,
                              'fit_peak_fwhm': None,
                              'full_fit_results': None})
        histogram_list[index].update({'peak_info': peak_info})
        # print(histogram_list)
        return

    def get_quadratic_calibration_values(self, histogram_dict, source_name):
        # finds centroid values of various sources
        index = 0
        for channel_dict_key in histogram_dict.keys():
            if sum(histogram_dict[channel_dict_key]) > 10000:
                self.source_channel_list.append({'channel': channel_dict_key, 'hist': histogram_dict[channel_dict_key]})

                # find the peaks of interest
                self.get_peak_guess(index, self.source_channel_list, source_name)
                self.find_peak_centroids(index, self.source_channel_list)
                index = index + 1
            else:
                print(f"  Channel {channel_dict_key} rejected due to insufficient counts.")

        # pprint(self.source_channel_list)
        return

    def get_peak_guess(self, index, histogram_list, source_name):
        # get linear calibration parameters and peak information
        cal_params = self.linear_cal_list[index]['poly_fit']
        self.get_peak_info(source_name)

        peak_info = []
        for peak in self.source_info['peak']:
            bin_guess = round((peak - cal_params[1]) / cal_params[0])
            peak_info.append({'peak_energy': peak,
                              'est_peak_centroid': bin_guess,
                              'fit_peak_centroid': None,
                              'fit_peak_fwhm': None,
                              'full_fit_results': None})
        histogram_list[index].update({'peak_info': peak_info})
        return

    def find_peak_centroids(self, index, histogram_list):
        channel_dict = histogram_list[index]
        bins = np.linspace(0, len(channel_dict['hist']), num=len(
            channel_dict['hist']), dtype=int)
        # define model for fitting
        model = Model(self.peak_function, independent_vars=['x'])
        for peak_index in range(len(channel_dict['peak_info'])):
            my_peak = channel_dict['peak_info'][peak_index]
            # give initial parameters
            params = Parameters()
            params.add('scale', value=5000)
            params.add('centroid', value=my_peak['est_peak_centroid'])
            params.add('sigma', value=1.0)
            params.add('linear_bg', value=5.0)
            params.add('scale_bg', value=100.0)

            # restricting range of fit
            x_min = my_peak['est_peak_centroid'] - 10
            x_max = my_peak['est_peak_centroid'] + 10
            counts_to_fit = channel_dict['hist'][x_min:x_max]
            bins_to_fit = bins[x_min:x_max]
            fit_results = model.fit(
                counts_to_fit, x=bins_to_fit, params=params)

            histogram_list[index]['peak_info'][peak_index]['fit_peak_centroid'] = fit_results.params['centroid'].value
            histogram_list[index]['peak_info'][peak_index]['fit_peak_fwhm'] = fit_results.params['sigma'].value
            histogram_list[index]['peak_info'][peak_index]['full_fit_results'] = fit_results

            # print(f"Channel: {channel_dict['channel']} Peak: {my_peak['peak_energy']}")
            # print(f"  Guess: {my_peak['est_peak_centroid']} Found: {my_peak['fit_peak_centroid']}")

        return

    def get_peak_info(self, source_name):
        if source_name.lower() == "60co":
            self.source_info['peak'] = [1173.228, 1332.492]
            self.source_info['peak_error'] = [0.003, 0.004]
            self.source_peak_finder_dict = {'prominence_fraction_of_tallest_peak': 1 / 3,
                                            'distance': 5,
                                            'num_peaks_needed': 2}
        elif source_name.lower() == "152eu":
            self.source_info['peak'] = [121.7817, 244.6974, 344.2785, 778.9045, 964.057, 1112.076, 1408.013]
        return

    def peak_function(self, x, scale, centroid, sigma, linear_bg, scale_bg):
        return scale * np.exp(-(np.power(x - centroid, 2) / np.power(2 * sigma, 2))) + linear_bg + scale_bg * (0.5 * (1 - (erf((x - centroid) / (sigma * np.power(2, 0.5))))))

    def write_calibration_file(self, cal_output_file):
        config = ConfigObj()
        config.filename = cal_output_file
        self.extend_cal_object(self.linear_cal_list, config)
        config.write()
        print(f'Writing calibration coefficients to: {cal_output_file}')
        return

    def extend_cal_object(self, hist_list, config):
        for my_hist in hist_list:
            config['CHAN' + str(my_hist['channel'])] = {}
            config['CHAN' + str(my_hist['channel'])]['channel'] = my_hist['channel']
            config['CHAN' + str(my_hist['channel'])]['cal_coeff'] = my_hist['poly_fit'].tolist()
        return

    def apply_calibration(self, data_df):
        self.read_in_calibration_file()
        # apply channel dependent energy calibration to entire dataframe
        data_df['energy'] = 0.0
        num_channels = 64
        for channel_key in self.cal_dict.keys():
            print(f'Calibrating channel: {channel_key} of {num_channels}', end='\r')
            data_df.loc[data_df.crystal == channel_key, 'energy'] = np.polyval(
                self.cal_dict[channel_key], data_df.loc[data_df['crystal'] == channel_key, 'charge'])
        return
