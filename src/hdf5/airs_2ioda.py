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

import numpy as np

from eccodes import *

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
    ("dateTime", "long", "seconds since 1970-01-01T00:00:00Z", "keep"),
]
meta_keys = [m_item[0] for m_item in locationKeyList]

iso8601_string = locationKeyList[meta_keys.index('dateTime')][2]
epoch = datetime.fromisoformat(iso8601_string[14:-1])


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


def get_data_from_files(afile):


    print(f"  ... afile: {afile}")
    f = open(afile, 'rb')
    bufr = codes_bufr_new_from_file(f)
    codes_set(bufr, 'unpack', 1)
    WMO_sat_ID = get_WMO_satellite_ID(afile)

    profile_meta_data = get_meta_data(bufr)
    obs_data = get_obs_data(bufr, profile_meta_data, add_qc, record_number=record_number)
    pdb.set_trace()


def get_meta_data(bufr):

    # get some of the global attributes that we are interested in
    meta_data_keys = def_meta_data()

    # these are the MetaData we are interested in
    profile_meta_data = {}
    for k, v in meta_data_keys.items():
        profile_meta_data[k] = codes_get(bufr, v)

    # do the hokey time structure to time structure
    year = codes_get(bufr, 'year')
    month = codes_get(bufr, 'month')
    day = codes_get(bufr, 'day')
    hour = codes_get(bufr, 'hour')
    minute = codes_get(bufr, 'minute')
    second = codes_get(bufr, 'second')  # non-integer value

    # should be able to import from atms or a def routine
    iterables = [year, month, day, hour, minute, second]
    # ensure the year is plausible (65535 appears in some data) if not set to 01Jan1900 (revisit)
    this_datetime = [datetime(adate[0], adate[1], adate[2], adate[3], adate[4], adate[5]) \
        if adate[0] < 2200 else datetime(1900,1,1,0,0,0) \
        for adate in zip(*iterables)]

    time_offset = [round((adatetime - epoch).total_seconds()) for adatetime in this_datetime]
    obs_data[('datetime', 'MetaData')] = np.array(get_epoch_time(g['obs_time_utc']), dtype='int64')

    profile_meta_data['dateTime'] = time_offset

    return profile_meta_data


def get_obs_data(bufr, profile_meta_data):

    # allocate space for output depending on which variables are to be saved
    obs_data = init_obs_loc()

    # replication factors for the datasets, bending angle, refractivity and derived profiles
    krepfac = codes_get_array(bufr, 'extendedDelayedDescriptorReplicationFactor')
    # array([247, 247, 200])

    drepfac = codes_get_array(bufr, 'delayedDescriptorReplicationFactor')
    # len(drepfac) Out[13]: 247   # ALL values all 3
    # sequence is 3 *(freq,impact,bendang,first-ord stat, bendang error, first-ord sat)
    #  note the label bendingAngle is used for both the value and its error !!!

    # get the bending angle
    lats = codes_get_array(bufr, 'latitude')[1:]                     # geolocation -- first value is the average
    lons = codes_get_array(bufr, 'longitude')[1:]
    bang = codes_get_array(bufr, 'bendingAngle')[4::drepfac[0]*2]    # L1, L2, combined -- only care about combined
    bang_err = codes_get_array(bufr, 'bendingAngle')[5::drepfac[0]*2]
    impact = codes_get_array(bufr, 'impactParameter')[2::drepfac[0]]
    bang_conf = codes_get_array(bufr, 'percentConfidence')[1:krepfac[0]+1]
    # len (bang) Out[19]: 1482  (krepfac * 6) -or- (krepfac * drepfac * 2 )`

    # bits are in reverse order according to WMO GNSSRO bufr documentation
    # ! Bit 1=Non-nominal quality
    # ! Bit 3=Rising Occulation (1=rising; 0=setting)
    # ! Bit 4=Excess Phase non-nominal
    # ! Bit 5=Bending Angle non-nominal
    i_non_nominal = get_normalized_bit(profile_meta_data['qualityFlag'], bit_index=16-1)
    i_phase_non_nominal = get_normalized_bit(profile_meta_data['qualityFlag'], bit_index=16-4)
    i_bang_non_nominal = get_normalized_bit(profile_meta_data['qualityFlag'], bit_index=16-5)
    iasc = get_normalized_bit(profile_meta_data['qualityFlag'], bit_index=16-3)
    # add rising/setting (ascending/descending) bit
    obs_data[('ascending_flag', 'MetaData')] = np.array(np.repeat(iasc, krepfac[0]), dtype=ioda_int_type)

    # print( " ... RO QC flags: %i  %i  %i  %i" % (i_non_nominal, i_phase_non_nominal, i_bang_non_nominal, iasc) )

    # exit if non-nominal profile
    if i_non_nominal != 0 or i_phase_non_nominal != 0 or i_bang_non_nominal != 0:
        return {}

    # value, ob_error, qc
    obs_data[('bending_angle', "ObsValue")] = assign_values(bang)
    obs_data[('bending_angle', "ObsError")] = assign_values(bang_err)
    obs_data[('bending_angle', "PreQC")] = np.full(krepfac[0], 0, dtype=ioda_int_type)

    # (geometric) height is read as integer but expected as float in output
    height = codes_get_array(bufr, 'height', ktype=float)

    # get the refractivity
    refrac = codes_get_array(bufr, 'atmosphericRefractivity')[0::2]
    refrac_err = codes_get_array(bufr, 'atmosphericRefractivity')[1::2]
    refrac_conf = codes_get_array(bufr, 'percentConfidence')[sum(krepfac[:1])+1:sum(krepfac[:2])+1]

    # value, ob_error, qc
    obs_data[('refractivity', "ObsValue")] = assign_values(refrac)
    obs_data[('refractivity', "ObsError")] = assign_values(refrac_err)
    obs_data[('refractivity', "PreQC")] = np.full(krepfac[0], 0, dtype=ioda_int_type)

    meta_data_types = def_meta_types()

    obs_data[('latitude', 'MetaData')] = assign_values(lats)
    obs_data[('longitude', 'MetaData')] = assign_values(lons)
    obs_data[('impact_parameter', 'MetaData')] = assign_values(impact)
    obs_data[('altitude', 'MetaData')] = assign_values(height)
    for k, v in profile_meta_data.items():
        if type(v) is int:
            obs_data[(k, 'MetaData')] = np.array(np.repeat(v, krepfac[0]), dtype=ioda_int_type)
        elif type(v) is float:
            obs_data[(k, 'MetaData')] = np.array(np.repeat(v, krepfac[0]), dtype=ioda_float_type)
        else:  # something else (datetime for instance)
            string_array = np.repeat(v.strftime("%Y-%m-%dT%H:%M:%SZ"), krepfac[0])
            obs_data[(k, 'MetaData')] = string_array.astype(object)

    # set record number (multi file procesing will change this)
    if record_number is None:
        nrec = 1
    else:
        nrec = record_number
    obs_data[('record_number', 'MetaData')] = np.array(np.repeat(nrec, krepfac[0]), dtype=ioda_int_type)

    # get derived profiles
    geop = codes_get_array(bufr, 'geopotentialHeight')[:-1]
    pres = codes_get_array(bufr, 'nonCoordinatePressure')[0:-2:2]
    temp = codes_get_array(bufr, 'airTemperature')[0::2]
    spchum = codes_get_array(bufr, 'specificHumidity')[0::2]
    prof_conf = codes_get_array(bufr, 'percentConfidence')[sum(krepfac[:2])+1:sum(krepfac)+1]

    # Compute impact height
    obs_data[('impact_height', 'MetaData')] = \
        obs_data[('impact_parameter', 'MetaData')] - \
        obs_data[('geoid_height_above_reference_ellipsoid', 'MetaData')] - \
        obs_data[('earth_radius_of_curvature', 'MetaData')]

    if add_qc:
        good = quality_control(profile_meta_data, height, lats, lons)
        if len(lats[good]) == 0:
            return{}
            # exit if entire profile is missing
        for k in obs_data.keys():
            obs_data[k] = obs_data[k][good]

    return obs_data


def get_WMO_satellite_ID(filename):

    afile = os.path.basename(filename)
    WMO_sat_ID = AQUA_WMO_sat_ID

    return WMO_sat_ID


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
        ('dateTime', 'MetaData'): [],
        ('scan_position', 'MetaData'): [],
        ('solar_zenith_angle', 'MetaData'): [],
        ('solar_azimuth_angle', 'MetaData'): [],
        ('sensor_zenith_angle', 'MetaData'): [],
        ('sensor_view_angle', 'MetaData'): [],
        ('sensor_azimuth_angle', 'MetaData'): [],
    }

    return obs


def def_meta_data():

    # keys in bufr file for single value metaData per footprint
    meta_data_keys = {
        "satelliteId", 'MetaData',
        "nchan", 'MetaData',
        "dateTime", 'MetaData',
    }

    return meta_data_keys


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
