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
import math
import numpy
import sys
import csv

class data_collection:
    def __init__(self, ral, psm_name, mtm_name, expected_interval):
        self.ral = ral
        self.expected_interval = expected_interval
        self.psm = dvrk.psm(ral = ral,
                            arm_name = psm_name,
                            expected_interval = expected_interval)
        
        self.mtm = dvrk.mtm(ral = ral,
                            arm_name = mtm_name,
                            expected_interval = expected_interval)

    def collect(self, csv_writer):
        rate = self.ral.create_rate(500)
        while True:
            data = []
            data.extend(self.psm.measured_jp())
            data.extend(self.psm.measured_jv())
            data.extend(self.psm.measured_jf())

            csv_writer.writerow(data)

            rate.sleep()

    def run(self):
        self.psm.check_connections()
        self.mtm.check_connections()

        with open('js_train.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            self.collect(writer)


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
    parser.add_argument('-i', '--interval', type=float, default=0.1,
                        help = 'expected interval in seconds between messages sent by the device')
    args = parser.parse_args(argv)

    ral = crtk.ral('data_collection')
    application = data_collection(ral, args.psm, args.mtm, args.interval)
    ral.spin_and_execute(application.run)
