import numpy as np
import scipy.signal
from configobj import ConfigObj
from lib.input_handler import input_handler
from lib.histogram_manager import histogram_manager
from lmfit import Model, Parameters
from lmfit.models import QuadraticModel
from scipy.special import erf

from pprint import pprint


class energy_calibration:

    def __init__(self, cal_file=None, max_energy=3000):
        self.cal_file = cal_file
        self.data_df = None
        self.cal_dict = {}
        self.max_energy = max_energy
        self.linear_cal_list = []
        self.quad_source_list = []
        self.quad_cal_data = []
        self.source_channel_list = []
        self.source_info = {}
        self.source_peak_finder_dict = {}

    def read_in_calibration_file(self):
        # read in calibration file and make a dict of coefficients
        print(f'Using calibration file: {self.cal_file}')
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
        self.data_df = my_input.read_in_data(source_dict['60co'])
        hist_man = histogram_manager(self.data_df)
        charge_hist_list = hist_man.generate_channel_histograms_1D('charge')
        self.perform_linear_calibration(charge_hist_list, '60co')
        # second perform quadratic calibration using other sources
        for source_key in source_dict.keys():
            if source_key == '60co':
                self.quad_source_list.append({'source': source_key, 'channel_info': self.linear_cal_list})
                continue
            else:
                print(f"Extracting {source_key} peak positions ...")
                self.data_df = my_input.read_in_data(source_dict[source_key])
                hist_man = histogram_manager(self.data_df)
                charge_hist_list = hist_man.generate_channel_histograms_1D('charge')
                self.get_quadratic_calibration_values(charge_hist_list, source_key)
                self.quad_source_list.append({'source': source_key, 'channel_info': self.source_channel_list})

        # now we perform the quadratic calibration
        self.perform_quadratic_calibration()
        # save histograms with labeled centroids
        self.write_diagnostic_histograms()
        return

    def perform_linear_calibration(self, histograms, source_name):
        # Creates initial 60Co linear calibration to make quadratic peak search easier
        print("Perfoming linear calibration ...")
        self.linear_cal_list = histograms
        for channel_index in range(len(histograms)):
            if sum(histograms[channel_index]['counts']) > 10000:
                # find the peaks of interest
                self.find_peaks(channel_index, self.linear_cal_list, source_name)
                self.find_peak_centroids(channel_index, self.linear_cal_list)
                # now we fit a linear function for basic linear calibration
                self.find_poly_fit(channel_index, self.linear_cal_list, 1)
            else:
                print(f"  Channel {histograms[channel_index]['channel']} rejected due to insufficient counts.")
                self.linear_cal_list[channel_index] = 0
        # write out calibration file
        self.linear_cal_list = list(filter(lambda entry: entry != 0, self.linear_cal_list))
        self.write_calibration_file('linear_calibration.cal', 'linear')
        return

    def perform_quadratic_calibration(self):
        # first we need to collect the peak data from all sources
        print("Perfoming quadratic calibration ...")
        self.combine_all_source_data()
        # write out guesses for diagnostics
        self.write_out_diagnostics('diagnostics/found_peaks.csv')
        model = QuadraticModel()
        param_list = []
        for channel_index in range(0, len(self.quad_cal_data)):
            energy_vals = np.array(self.quad_cal_data[channel_index]['energy'])
            energy_error_vals = np.array(self.quad_cal_data[channel_index]['energy_error'])
            centroid_vals = np.array(self.quad_cal_data[channel_index]['centroids'])
            fit = model.fit(energy_vals, x=centroid_vals, a=1.0, b=1.0, c=1.0)
            #fit = model.fit(energy_vals, x=centroid_vals, a=1.0, b=1.0, c=1.0, weights=1.0 / energy_error_vals, scale_covar=False)
            for key in fit.params:
                param_list.append(fit.params[key].value)
            self.quad_cal_data[channel_index].update({'poly_fit': param_list})
            param_list = []

            #poly_fit = np.polyfit(self.quad_cal_data[channel_index]['centroids'], self.quad_cal_data[channel_index]['energy'], 2)
        # write out calibration file
        self.write_calibration_file('quadratic_calibration.cal', 'quadratic')
        return

    def combine_all_source_data(self):
        # first aggregate the channel data from all sources used
        self.quad_cal_data = [0] * 64
        for my_source_index in range(len(self.quad_source_list)):
            my_source = self.quad_source_list[my_source_index]['channel_info']
            index = 0
            for my_channel_index in range(len(my_source)):
                my_peak_info = my_source[my_channel_index]['peak_info']
                if self.quad_cal_data[index] == 0:
                    self.quad_cal_data[index] = {'channel': my_source[my_channel_index]['channel'], 'energy': [], 'centroids': [], 'energy_error': []}
                # extract fitted peak centroids
                for my_peak_index in range(len(my_peak_info)):
                    self.quad_cal_data[index]['energy'].append(my_peak_info[my_peak_index]['peak_energy'])
                    self.quad_cal_data[index]['centroids'].append(my_peak_info[my_peak_index]['fit_peak_centroid'])
                    self.quad_cal_data[index]['energy_error'].append(my_peak_info[my_peak_index]['peak_energy_error'])
                index = index + 1
        # pprint(self.quad_cal_data)
        # remove zero channels
        self.quad_cal_data = list(filter(lambda entry: entry != 0, self.quad_cal_data))
        return

    def write_out_diagnostics(self, filename):
        print(f'Writing peak energies/guesses to file: {filename}')
        diagnostics_file = open(filename, 'w')
        diagnostics_file.write('channel, energy, centoid_guess\n')
        for channel in self.quad_cal_data:
            for energy_index in range(len(channel['energy'])):
                diagnostics_file.write(f"{channel['channel']}, {channel['energy'][energy_index]}, {channel['centroids'][energy_index]}\n")
        diagnostics_file.close()
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
        tallest_peak = np.amax(histogram_dict['counts'])
        tallest_peak_index = np.where(histogram_dict['counts'] == tallest_peak)[0][0]
        prominence = tallest_peak * self.source_peak_finder_dict['prominence_fraction_of_tallest_peak']
        peak_indices, peak_properties = scipy.signal.find_peaks(histogram_dict['counts'], prominence=prominence)

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
                              'peak_energy_error': self.source_info['peak_error'][my_peak_index],
                              'est_peak_centroid': peak_indices[my_peak_index],
                              'fit_peak_centroid': None,
                              'fit_peak_fwhm': None,
                              'full_fit_results': None})
        histogram_list[index].update({'peak_info': peak_info})
        # print(histogram_list)
        return

    def get_quadratic_calibration_values(self, histogram_list, source_name):
        # finds centroid values of various sources
        self.source_channel_list = histogram_list
        for channel_index in range(len(histogram_list)):
            if sum(histogram_list[channel_index]['counts']) > 10000:
                # find the peaks of interest
                #print(f"Channel: {histogram_list[channel_index]['channel']} Index: {channel_index}")
                self.get_peak_guess(channel_index, histogram_list[channel_index]['channel'], source_name)
                self.find_peak_centroids(channel_index, self.source_channel_list)
            else:
                print(f"  Channel {histogram_list[channel_index]['channel']} rejected due to insufficient counts.")
                self.source_channel_list[channel_index] = 0

        # remove zero channels
        self.source_channel_list = list(filter(lambda entry: entry != 0, self.source_channel_list))
        return

    def get_peak_guess(self, index, channel, source_name):
        # get linear calibration parameters and peak information
        self.get_peak_info(source_name)
        cal_params = self.get_calibration_parameters(channel, 'linear')
        peak_info = []
        for peak_index in range(len(self.source_info['peak'])):
            peak = self.source_info['peak'][peak_index]
            bin_guess = round((peak - cal_params[1]) / cal_params[0])
            peak_info.append({'peak_energy': peak,
                              'peak_energy_error': self.source_info['peak_error'][peak_index],
                              'est_peak_centroid': bin_guess,
                              'fit_peak_centroid': None,
                              'fit_peak_fwhm': None,
                              'full_fit_results': None})
        self.source_channel_list[index].update({'peak_info': peak_info})
        return

    def get_calibration_parameters(self, channel, cal_type):
        if cal_type == 'linear':
            param_list = self.linear_cal_list
        elif cal_type == 'quadratic':
            print("need to implement")
            return

        # find the proper channel calibration
        params = next((channel_dict['poly_fit'] for channel_dict in param_list if channel_dict['channel'] == channel), None)
        return params

    def find_peak_centroids(self, index, histogram_list):
        channel_dict = histogram_list[index]
        # bins = np.linspace(0, len(channel_dict['hist']), num=len(channel_dict['hist']), dtype=int)
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
            x_min = my_peak['est_peak_centroid'] - 8
            x_max = my_peak['est_peak_centroid'] + 8
            counts_to_fit = channel_dict['counts'][x_min:x_max]
            counts_to_fit_error = channel_dict['error'][x_min:x_max]
            bins_to_fit = channel_dict['bins'][x_min:x_max]
            # replace any zeros in errors
            counts_to_fit_error[counts_to_fit_error == 0.] = 1.0

            fit_results = model.fit(counts_to_fit, x=bins_to_fit, params=params, weights=1.0 / counts_to_fit_error, scale_covar=False)

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
            #self.source_info['peak'] = [778.9045, 964.057, 1112.076, 1408.013]
            #self.source_info['peak_error'] = [0.0024, 0.005, 0.003, 0.003]
            self.source_info['peak'] = [121.77, 244.6974, 344.2785, 443.9606, 778.9045, 964.057, 1112.076, 1408.013]
            self.source_info['peak_error'] = [0.03, 0.0008, 0.0012, 0.0016, 0.0024, 0.005, 0.003, 0.003]
        elif source_name.lower() == "133ba":
            self.source_info['peak'] = [276.3989, 302.8508, 356.0129, 383.8485]
            self.source_info['peak_error'] = [0.0012, 0.0005, 0.0007, 0.0012]
        elif source_name.lower() == "207bi":
            self.source_info['peak'] = [569.698, 1063.656, 1770.228]
            self.source_info['peak_error'] = [0.002, 0.003, 0.009]
        return

    def peak_function(self, x, scale, centroid, sigma, linear_bg, scale_bg):
        return scale * np.exp(-(np.power(x - centroid, 2) / (2 * np.power(sigma, 2)))) + linear_bg + scale_bg * (0.5 * (1 - (erf((x - centroid) / (sigma * np.power(2, 0.5))))))

    def write_diagnostic_histograms(self):
        # saves channel histograms with found centroids labeled
        print("Saving histograms ... ")
        my_hist_man = histogram_manager(None)
        for my_source_index in range(len(self.quad_source_list)):
            my_source = self.quad_source_list[my_source_index]['channel_info']
            my_source_name = self.quad_source_list[my_source_index]['source']
            for my_channel_index in range(len(my_source)):
                my_channel_name = my_source[my_channel_index]['channel']
                my_peak_info = my_source[my_channel_index]['peak_info']
                my_hist_man.draw_histogram(my_source[my_channel_index]['counts'], my_source[my_channel_index]['bins'], f'histograms/{my_source_name}.{my_channel_name}.png', f'{my_source_name} channel:{my_channel_name}', peak_labels=my_peak_info)
        return


    def write_calibration_file(self, cal_output_file, calibration_type):
        config = ConfigObj()
        config.filename = cal_output_file
        if calibration_type == 'linear':
            self.extend_cal_object(self.linear_cal_list, config, linear=True)
        elif calibration_type == 'quadratic':
            self.extend_cal_object(self.quad_cal_data, config)
        else:
            print("No calibration type specified, exiting")
            exit(1)
        config.write()
        print(f'Writing calibration coefficients to: {cal_output_file}')
        return

    def extend_cal_object(self, hist_list, config, linear=False):
        for my_hist in hist_list:
            config['CHAN' + str(my_hist['channel'])] = {}
            config['CHAN' + str(my_hist['channel'])]['channel'] = my_hist['channel']
            if linear:
                config['CHAN' + str(my_hist['channel'])]['cal_coeff'] = my_hist['poly_fit'].tolist()
            else:
                config['CHAN' + str(my_hist['channel'])]['cal_coeff'] = my_hist['poly_fit']
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
