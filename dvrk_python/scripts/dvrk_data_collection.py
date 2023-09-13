#!/usr/bin/env python

# Author: Anton Deguet
# Date: 2015-02-22

# (C) Copyright 2015-2023 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import argparse
import crtk
import dvrk
import sys
import csv

class clutch_offset:
    def __init__(self, ral, expected_interval):
        self._crtk_utils = crtk.utils(self, ral, expected_interval)
        self._crtk_utils.add_measured_cp()
        self._ral = ral

class force_sensor:
    def __init__(self, ral, expected_interval):
        self._crtk_utils = crtk.utils(self, ral, expected_interval)
        self._crtk_utils.add_measured_cf()
        self._ral = ral

class data_collection:
    def __init__(self, ral, psm_name, mtm_name, output_file, expected_interval):
        self.ral = ral
        self.expected_interval = expected_interval
        self.output_file = output_file
        self.psm = dvrk.psm(ral = ral,
                            arm_name = psm_name,
                            expected_interval = expected_interval)
        
        self.mtm = dvrk.mtm(ral = ral,
                            arm_name = mtm_name,
                            expected_interval = expected_interval)
        
        self.psm_initial = clutch_offset(ral.create_child("MTML_PSM2/PSM/initial"), expected_interval)
        self.mtm_initial = clutch_offset(ral.create_child("MTML_PSM2/MTM/initial"), expected_interval)
        self.force_sensor = force_sensor(ral.create_child("force_sensor"), expected_interval)

    def collect(self, csv_writer):
        rate = self.ral.create_rate(500)

        print("Collecting data...")
        while not self.ral.is_shutdown() and self.psm.is_enabled():
            data = []

            data.append(self.psm.measured_cp().p[0] - self.psm_initial.measured_cp().p[0])
            data.append(self.psm.measured_cp().p[1] - self.psm_initial.measured_cp().p[1])
            data.append(self.psm.measured_cp().p[2] - self.psm_initial.measured_cp().p[2])

            data.append(self.mtm.measured_cp().p[0] - self.mtm_initial.measured_cp().p[0])
            data.append(self.mtm.measured_cp().p[1] - self.mtm_initial.measured_cp().p[1])
            data.append(self.mtm.measured_cp().p[2] - self.mtm_initial.measured_cp().p[2])

            data.extend(self.psm.measured_cp().M.GetQuaternion())
            data.extend(self.mtm.measured_cp().M.GetQuaternion())

            data.extend(self.psm.external.measured_cf())
            data.extend(self.mtm.external.measured_cf())

            data.extend(self.force_sensor.measured_cf())

            csv_writer.writerow(data)

            rate.sleep()

    def run(self):
        self.ral.check_connections()

        try:
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                self.collect(writer)
        except:
            return


if __name__ == '__main__':
    argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--psm', type=str, required=True,
                        choices=['PSM1', 'PSM2', 'PSM3'],
                        help = 'arm name corresponding to ROS topics without namespace.  Use __ns:= to specify the namespace')
    parser.add_argument('-m', '--mtm', type=str, required=True,
                        choices=['MTML', 'MTMR'],
                        help = 'arm name corresponding to ROS topics without namespace.  Use __ns:= to specify the namespace')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help = 'output file name')
    parser.add_argument('-i', '--interval', type=float, default=0.1,
                        help = 'expected interval in seconds between messages sent by the device')
    args = parser.parse_args(argv)

    ral = crtk.ral('data_collection')
    application = data_collection(ral, args.psm, args.mtm, args.output, args.interval)
    ral.spin_and_execute(application.run)
