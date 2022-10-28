#!/usr/bin/python

"""
Python code to ingest netCDF4 or HDF5 ATMS data
"""

import argparse
from datetime import datetime, timedelta
import glob
# from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os.path
from os import getcwd
import sys
import pdb

import h5py
import numpy as np

from atms_netcdf_hdf5_2ioda import write_obs_2ioda

IODA_CONV_PATH = Path(__file__).parent/"@SCRIPT_LIB_PATH@"
if not IODA_CONV_PATH.is_dir():
    IODA_CONV_PATH = Path(__file__).parent/'..'/'lib-python'
sys.path.append(str(IODA_CONV_PATH.resolve()))
import ioda_conv_engines as iconv
from orddicts import DefaultOrderedDict

# globals
AQUA_WMO_sat_ID = 79

# and more globals
missing_value = 9.96921e+36
int_missing_value = -2147483647

GlobalAttrs = {
    "platformCommonName": "ATMS",
    "platformLongDescription": "ATMS Brightness Temperature Data",
    "sensorCentralFrequency": [23.8,
                               31.4, 50.3, 51.76, 52.8, 53.596, 54.40, 54.94, 55.50,
                               57.2903, 57.2903, 57.2903, 57.2903, 57.2903, 57.2903,
                               88.20, 165.5, 183.31, 183.31, 183.31, 183.31, 183.31],
}

locationKeyList = [
    ("latitude", "float"),
    ("longitude", "float"),
    ("datetime", "string")
]


def main(args):

    output_filename = args.output
    dtg = datetime.strptime(args.date, '%Y%m%d%H')

    input_files = [(i) for i in args.input]
    # read / process files in parallel
    obs_data = {}
    # create a thread pool
#   with ProcessPoolExecutor(max_workers=args.threads) as executor:
#       for file_obs_data in executor.map(get_data_from_files, input_files):
#           if not file_obs_data:
#               print("INFO: non-nominal file skipping")
#               continue
#           if obs_data:
#               concat_obs_dict(obs_data, file_obs_data)
#           else:
#               obs_data = file_obs_data

    for afile in input_files:
        file_obs_data = get_data_from_files(afile)
        if not file_obs_data:
            print("INFO: non-nominal file skipping")
            continue
        if obs_data:
            concat_obs_dict(obs_data, file_obs_data)
        else:
            obs_data = file_obs_data

    GlobalAttrs = get_global_attributes('airs')

    write_obs_2ioda(obs_data, GlobalAttrs)


def get_data_from_files(zfiles):

    # allocate space for output depending on which variables are to be saved
    obs_data = init_obs_loc()

    # for afile in zfiles:
    afile = zfiles
    if True:
        f = h5py.File(afile, 'r')
        obs_data = get_data(f, g, obs_data)
        f.close()

    return obs_data


def get_data(f, obs_data):

    # NASA GES DISC keys
    # 'antenna', 'antenna_len', 'antenna_temp', 'antenna_temp_qc', 'asc_flag', 'asc_node_local_solar_time',
    # 'asc_node_lon', 'asc_node_tai93', 'atrack', 'attitude', 'attitude_lbl', 'attitude_lbl_len',
    # 'aux_cal_blackbody_qualflag', 'aux_cal_qualflag', 'aux_cal_space_qualflag', 'aux_cold_temp',
    # 'aux_gain', 'aux_geo_qualflag', 'aux_nonlin', 'aux_offset', 'aux_warm_temp', 'band', 'band_geoloc_chan',
    # 'band_land_frac', 'band_lat', 'band_lat_bnds', 'band_lbl', 'band_lbl_len', 'band_lon', 'band_lon_bnds',
    # 'band_surf_alt', 'bandwidth', 'beam_width', 'center_freq', 'chan_band', 'chan_band_len', 'channel',
    # 'cold_nedt', 'fov_poly', 'if_offset_1', 'if_offset_2', 'instrument_state', 'land_frac', 'lat',
    # 'lat_bnds', 'lat_geoid', 'local_solar_time', 'lon', 'lon_bnds', 'lon_geoid', 'mean_anom_wrt_equat',
    # 'moon_ang', 'obs_id', 'obs_id_len', 'obs_time_tai93', 'obs_time_utc', 'polarization', 'polarization_len',
    # 'sat_alt', 'sat_att', 'sat_azi', 'sat_pos', 'sat_range', 'sat_sol_azi', 'sat_sol_zen', 'sat_vel', 'sat_zen',
    # 'scan_mid_time', 'sol_azi', 'sol_zen', 'solar_beta_angle', 'spacextrack', 'spatial', 'spatial_lbl',
    # 'spatial_lbl_len', 'subsat_lat', 'subsat_lon', 'sun_glint_dist', 'sun_glint_lat', 'sun_glint_lon',
    # 'surf_alt', 'surf_alt_sdev', 'utc_tuple', 'utc_tuple_lbl', 'utc_tuple_lbl_len', 'view_ang', 'warm_nedt', 'xtrack'

    pdb.set_trace()
    WMO_sat_ID = get_WMO_satellite_ID(f.filename)

    # example: dimension ( 180, 96 ) == dimension( nscan, nbeam_pos )
    try:
        nscans = np.shape(g['lat'])[0]
        nbeam_pos = np.shape(g['lat'])[1]
        obs_data[('latitude', 'MetaData')] = np.array(g['lat'][:, :].flatten(), dtype='float32')
        obs_data[('longitude', 'MetaData')] = np.array(g['lon'][:, :].flatten(), dtype='float32')
        obs_data[('channelNumber', 'MetaData')] = np.array(g['channel'][:], dtype='int32')
# V2     obs_data[('fieldOfViewNumber', 'MetaData')] = np.tile(np.arange(nbeam_pos, dtype='int32') + 1, (nscans, 1)).flatten()
        obs_data[('scan_position', 'MetaData')] = np.tile(np.arange(nbeam_pos, dtype='float32') + 1, (nscans, 1)).flatten()
# V2     obs_data[('solarZenithAngle', 'MetaData')] = np.array(g['sol_zen'][:, :].flatten(), dtype='float32')
        obs_data[('solar_zenith_angle', 'MetaData')] = np.array(g['sol_zen'][:, :].flatten(), dtype='float32')
# V2     obs_data[('solarAzimuthAngle', 'MetaData')] = np.array(g['sol_azi'][:, :].flatten(), dtype='float32')
        obs_data[('solar_azimuth_angle', 'MetaData')] = np.array(g['sol_azi'][:, :].flatten(), dtype='float32')
# V2     obs_data[('sensorZenithAngle', 'MetaData')] = np.array(g['sat_zen'][:, :].flatten(), dtype='float32')
        obs_data[('sensor_zenith_angle', 'MetaData')] = np.array(g['sat_zen'][:, :].flatten(), dtype='float32')
# V2     obs_data[('sensorAzimuthAngle', 'MetaData')] = np.array(g['sat_azi'][:, :].flatten(), dtype='float32')
        obs_data[('sensor_azimuth_angle', 'MetaData')] = np.array(g['sat_azi'][:, :].flatten(), dtype='float32')
        obs_data[('sensor_view_angle', 'MetaData')] = np.array(g['view_ang'][:, :].flatten(), dtype='float32')
        nlocs = len(obs_data[('latitude', 'MetaData')])
        obs_data[('satelliteId', 'MetaData')] = np.full((nlocs), WMO_sat_ID, dtype='int32')
        obs_data[('datetime', 'MetaData')] = np.array(get_string_dtg(g['obs_time_utc'][:, :, :]), dtype=object)

    # BaseException is a catch-all mechamism
    except BaseException:
        # this section is for the NOAA CLASS files and need to be tested
        obs_data[('latitude', 'MetaData')] = np.array(g['All_Data']['ATMS-SDR-GEO_All']['Latitude'][:, :].flatten(), dtype='float32')
        obs_data[('longitude', 'MetaData')] = np.array(g['All_Data']['ATMS-SDR-GEO_All']['Longitude'][:, :].flatten(), dtype='float32')

    # example: dimension ( 180, 96, 22 ) == dimension( nscan, nbeam_pos, nchannel )
    try:
        nchans = len(obs_data[('channelNumber', 'MetaData')])
        nlocs = len(obs_data[('latitude', 'MetaData')])
# V2     obs_data[('brightnessTemperature', "ObsValue")] = np.array(np.vstack(g['antenna_temp']), dtype='float32')
# V2     obs_data[('brightnessTemperature', "ObsError")] = np.full((nlocs, nchans), 5.0, dtype='float32')
# V2     obs_data[('brightnessTemperature', "PreQC")] = np.full((nlocs, nchans), 0, dtype='int32')
        obs_data[('brightness_temperature', "ObsValue")] = np.array(np.vstack(g['antenna_temp']), dtype='float32')
        obs_data[('brightness_temperature', "ObsError")] = np.full((nlocs, nchans), 5.0, dtype='float32')
        obs_data[('brightness_temperature', "PreQC")] = np.full((nlocs, nchans), 0, dtype='int32')
    except BaseException:
        # this section is for the NOAA CLASS files and need to be tested
        scaled_data = np.vstack(f['All_Data']['ATMS-SDR_All']['BrightnessTemperature'])
        scale_fac = f['All_Data']['ATMS-SDR_All']['BrightnessTemperatureFactors'][:].flatten()

        obs_data[('brightnessTemperature', "ObsValue")] = np.array((scaled_data * scale_fac[0]) + scale_fac[1], dtype='float32')
        obs_data[('brightnessTemperature', "ObsError")] = np.full((nlocs, nchans), 5.0, dtype='float32')
        obs_data[('brightnessTemperature', "PreQC")] = np.full((nlocs, nchans), 0, dtype='int32')

    return obs_data


def get_WMO_satellite_ID(filename):

    afile = os.path.basename(filename)
    WMO_sat_ID = AQUA_WMO_sat_ID

    return WMO_sat_ID


def get_string_dtg(obs_time_utc):

    year = obs_time_utc[:, :, 0].flatten()
    month = obs_time_utc[:, :, 1].flatten()
    day = obs_time_utc[:, :, 2].flatten()
    hour = obs_time_utc[:, :, 3].flatten()
    minute = obs_time_utc[:, :, 4].flatten()
    dtg = []
    for i, yyyy in enumerate(year):
        cdtg = ("%4i-%.2i-%.2iT%.2i:%.2i:00Z" % (yyyy, month[i], day[i], hour[i], minute[i]))
        dtg.append(cdtg)

    return dtg


def init_obs_loc():
    # V2     ('brightnessTemperature', "ObsValue"): [],
    # V2     ('brightnessTemperature', "ObsError"): [],
    # V2     ('brightnessTemperature', "PreQC"): [],
    # V2     ('fieldOfViewNumber', 'MetaData'): [],
    # V2     ('solarZenithAngle', 'MetaData'): [],
    # V2     ('solarAzimuthAngle', 'MetaData'): [],
    # V2     ('sensorZenithAngle', 'MetaData'): [],
    # V2     ('sensorAzimuthAngle', 'MetaData'): [],
    obs = {
        ('brightness_temperature', "ObsValue"): [],
        ('brightness_temperature', "ObsError"): [],
        ('brightness_temperature', "PreQC"): [],
        ('satelliteId', 'MetaData'): [],
        ('channelNumber', 'MetaData'): [],
        ('latitude', 'MetaData'): [],
        ('longitude', 'MetaData'): [],
        ('datetime', 'MetaData'): [],
        ('scan_position', 'MetaData'): [],
        ('solar_zenith_angle', 'MetaData'): [],
        ('solar_azimuth_angle', 'MetaData'): [],
        ('sensor_zenith_angle', 'MetaData'): [],
        ('sensor_view_angle', 'MetaData'): [],
        ('sensor_azimuth_angle', 'MetaData'): [],
    }

    return obs


def concat_obs_dict(obs_data, append_obs_data):
    # For now we are assuming that the obs_data dictionary has the "golden" list
    # of variables. If one is missing from append_obs_data, a warning will be issued.
    append_keys = list(append_obs_data.keys())
    for gv_key in obs_data.keys():
        if gv_key in append_keys:
            obs_data[gv_key] = np.append(obs_data[gv_key], append_obs_data[gv_key], axis=0)
        else:
            print("WARNING: ", gv_key, " is missing from append_obs_data dictionary")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            'Reads the satellite data '
            ' convert into IODA formatted output files. '
            ' Multiple files are concatenated')
    )

    required = parser.add_argument_group(title='required arguments')
    required.add_argument(
        '-i', '--input',
        help="path of satellite observation input file(s)",
        type=str, nargs='+', required=True)
    required.add_argument(
        '-d', '--date',
        metavar="YYYYMMDDHH",
        help="base date for the center of the window",
        type=str, required=True)

    optional = parser.add_argument_group(title='optional arguments')
    optional.add_argument(
        '-j', '--threads',
        help='multiple threads can be used to load input files in parallel.'
             ' (default: %(default)s)',
        type=int, default=1)
    optional.add_argument(
        '-o', '--output',
        help='path to output ioda file',
        type=str, default=os.getcwd())

    args = parser.parse_args()

    main(args)
