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
import matplotlib.pyplot as plt
import sys
import csv

class plot:
    def __init__(self, ax, color, dashed, data_acquire, history):
        line_format = color + '+' if dashed else color + '-'
        plots = ax.plot([], [], line_format)
        self.plot = plots[0]

        self.data_acquire = data_acquire
        self.history = history
        self.data = []

    def poll(self):
        self.data.append(self.data_acquire())
        self.data = self.data[-self.history:]
        
    def update(self, xdata):
        self.plot.set_ydata(self.data)
        self.plot.set_xdata(xdata)


class data_plotter:
    def __init__(self, ral, psm_name, mtm_name, expected_interval):
        self.ral = ral
        self.expected_interval = expected_interval
        self.psm = dvrk.psm(ral = ral,
                            arm_name = psm_name,
                            expected_interval = expected_interval)
        
        self.mtm = dvrk.mtm(ral = ral,
                            arm_name = mtm_name,
                            expected_interval = expected_interval)

    def run(self):
        self.psm.check_connections()
        self.mtm.check_connections()

        plt.ion()
        fig, ax = plt.subplots()
        history_size = 30

        t = []
        plots = []
        plots.append(plot(ax, 'r', False, lambda: self.psm.spatial.measured_cf()[0], history_size))
        plots.append(plot(ax, 'r', True, lambda: 0.2 * self.mtm.spatial.measured_cf()[0], history_size))
        plots.append(plot(ax, 'g', False, lambda: self.psm.spatial.measured_cf()[1], history_size))
        plots.append(plot(ax, 'g', True, lambda: 0.2 * self.mtm.spatial.measured_cf()[1], history_size))
        plots.append(plot(ax, 'b', False, lambda: self.psm.spatial.measured_cf()[2], history_size))
        plots.append(plot(ax, 'b', True, lambda: 0.2 * self.mtm.spatial.measured_cf()[2], history_size))

        self.ral.create_rate(5).sleep()

        start = self.ral.now().to_sec()
        time = lambda: self.ral.now().to_sec() - start
        
        rate = 200
        ros_rate = self.ral.create_rate(rate)
        c = 0
        while True:
            t.append(time())

            if (len(t) > history_size):
                t = t[-history_size:]

            for p in plots:
                p.poll()

            for p in plots:
                p.update(t)

            ax.autoscale()
            ax.relim()
            bottom, top = ax.get_ylim()
            min_scale = 0.01
            if top - bottom < min_scale:
                avg = 0.5 * (top + bottom)
                ax.set_ylim(avg - 0.5*min_scale, avg + 0.5*min_scale)

            ax.autoscale_view()

            plt.pause(1.0/(0.2*rate))
            plt.draw()
            ros_rate.sleep()


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
    parser.add_argument('-i', '--interval', type=float, default=0.01,
                        help = 'expected interval in seconds between messages sent by the device')
    args = parser.parse_args(argv)

    ral = crtk.ral('data_plotter')
    application = data_plotter(ral, args.psm, args.mtm, args.interval)
    ral.spin_and_execute(application.run)
