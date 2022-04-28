#!/usr/bin/env python3

#
# (C) Copyright 2019 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#

from __future__ import print_function
import sys
import argparse
import netCDF4 as nc
from datetime import datetime, timedelta
import dateutil.parser
import numpy as np
from pathlib import Path

IODA_CONV_PATH = Path(__file__).parent/"@SCRIPT_LIB_PATH@"
if not IODA_CONV_PATH.is_dir():
    IODA_CONV_PATH = Path(__file__).parent/'..'/'lib-python'
sys.path.append(str(IODA_CONV_PATH.resolve()))

from orddicts import DefaultOrderedDict
import ioda_conv_engines as iconv


class Observation(object):

    def __init__(self, filename, date, pole):

        self.filename = filename
        self.date = date
        self.data = DefaultOrderedDict(lambda: DefaultOrderedDict(dict))
        self.pole = pole
        self._read(date)

    def _read(self, date):
        ncd = nc.MFDataset(self.filename)
        datein = ncd.variables['dtg_yyyymmdd'][:]
        timein = ncd.variables['dtg_hhmm'][:]
        lons = ncd.variables['longitude'][:]
        lats = ncd.variables['latitude'][:]
        vals = ncd.variables['ice_concentration'][:]
        qc = ncd.variables['quality'][:]
        ncd.close()

        if self.pole == 'north':
            goodobs = np.where( ((qc == 1) | (qc == 4)) & (lats>45.0) )
        if self.pole == 'south':
            goodobs = np.where( ((qc == 1) | (qc == 4)) & (lats<-40.0) )

        datein = datein[goodobs]
        timein = timein[goodobs]
        lons = lons[goodobs]
        lats = lats[goodobs]
        vals = vals[goodobs]
        qc = qc[goodobs]

        valKey = vName, iconv.OvalName()
        errKey = vName, iconv.OerrName()
        qcKey = vName, iconv.OqcName()
        date2 = int(date.strftime("%Y%m%d"))
        for i in range(len(lons)):
            if datein[i] == date2:
                obs_date = datetime.combine(
                    datetime.strptime(
                        np.array2string(
                            datein[i]), "%Y%m%d"), datetime.strptime(
                        np.array2string(
                            timein[i]).zfill(4), "%H%M").time())
                locKey = lats[i], lons[i], obs_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                self.data[locKey][valKey] = vals[i]
                self.data[locKey][errKey] = 0.1
                self.data[locKey][qcKey] = 0

vName = "sea_ice_area_fraction"

locationKeyList = [
    ("latitude", "float"),
    ("longitude", "float"),
    ("datetime", "string")
]

GlobalAttrs = {
    'odb_version': 1,
}


def main():

    parser = argparse.ArgumentParser(
        description=('')
    )

    required = parser.add_argument_group(title='required arguments')
    required.add_argument(
        '-i', '--input',
        help="EMC ice fraction obs input file(s)",
        type=str, nargs='+', required=True)
    required.add_argument(
        '-o', '--output',
        help="name of ioda output file",
        type=str, required=True)
    required.add_argument(
        '-d', '--date',
        help="base date for the center of the window",
        metavar="YYYYMMDDHH", type=str, required=True)
    required.add_argument(
        '-p', '--pole',
        help="north or south",
        type=str, required=True)

    optional = parser.add_argument_group(title='optional arguments')

    args = parser.parse_args()
    fdate = datetime.strptime(args.date, '%Y%m%d%H')
    VarDims = {
        vName: ['nlocs'],
    }
    # Read in
    ice = Observation(args.input, fdate, args.pole)

    # write them out
    ObsVars, nlocs = iconv.ExtractObsData(ice.data, locationKeyList)
    DimDict = {'nlocs': nlocs}
    writer = iconv.IodaWriter(args.output, locationKeyList, DimDict)

    VarAttrs = DefaultOrderedDict(lambda: DefaultOrderedDict(dict))
    VarAttrs[vName, iconv.OvalName()]['units'] = '1'
    VarAttrs[vName, iconv.OerrName()]['units'] = '1'
    VarAttrs[vName, iconv.OqcName()]['units'] = 'unitless'

    writer.BuildIoda(ObsVars, VarDims, VarAttrs, GlobalAttrs)


if __name__ == '__main__':
    main()
