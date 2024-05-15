#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

"""
        DAB Modulation Toolbox
        Copyright (C) 2023  strayedelectron

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Affero General Public License as
        published by the Free Software Foundation, either version 3 of the
        License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Affero General Public License for more details.

        You should have received a copy of the GNU Affero General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import serial
import struct
import copy

import numpy as np
import math
from datetime import datetime
# import logging
import argparse

import dab_datasets as ds
import debug_tools as db
from debug_tools import *
import plot_dab

# values we want to measure
# l_meas_keys = ['P_dc1', 'P_dc2', 'I_HF1', 'v_ds_S11_sw_on', 'v_ds_S23_sw_on']
# l_meas_keys = ['P_dc1', 'P_dc2', 'I_HF1', 'i_HF1_S11_sw_on', 'v_ds_S11_sw_on']
l_meas_keys = ['I_dc1', 'I_dc2', 'I_HF1', 'i_HF1_S11_sw_on', 'v_ds_S11_sw_on']
meas_prefix = 'meas_'


class DAB_Control():
    """
    Class to control the DAB over serial.
    """
    dab = None
    log = None
    ser = None
    directory = str()
    # States
    STATE_RUN = 0xAA
    STATE_STOP = 0xEE
    baud = 115200
    # Set default names
    name = 'lab_measurement'
    comment = 'DAB laboratory measurements.'

    def __init__(self, filename=str(), directory=str(), serial_port='/dev/ttyUSB0'):
        """
        Create control object that holds the DAB data and saves measurement data.
        :param filename: npz filepath that contains a DAB Dataset
        :param directory: output directory for the results and log
        """

        ## Expand relative paths to absolute
        filename = os.path.expanduser(filename)
        filename = os.path.expandvars(filename)
        filename = os.path.abspath(filename)
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)

        ## TEST
        # print(os.path.basename(filename))
        # sys.exit(2)

        ## Load DAB Dataset
        if not os.path.isfile(filename):
            print('DAB file does not exist: ' + filename)
            sys.exit(1)
        else:
            self.dab = ds.load_from_file(filename)

        ## Saving
        # Create new dir for all files
        if not os.path.exists(directory):
            print('Output directory does not exist: ' + directory)
            sys.exit(1)
        else:
            directory = os.path.join(directory,
                                     datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + self.name
                                     + "__" + os.path.basename(filename))
            directory = os.path.expanduser(directory)
            directory = os.path.expandvars(directory)
            directory = os.path.abspath(directory)
            os.mkdir(directory)
            self.directory = directory
            # Logging
            self.log = db.log(filename=os.path.join(directory, 'dab_control.log'))

        ## Init serial port
        self.ser = serial.Serial(serial_port, self.baud, timeout=1)

        self.comment += self.comment + ' Used npz file: ' + os.path.basename(filename) + ' ' + str(self.dab._comment)

    def __del__(self):
        print('Closing...')
        # self.save()
        try:
            self.serial_STOP()
            self.ser.close()
        except:
            print('Please check if everything is in secure state.')

    def __exit__(self):
        print('Exiting...')
        # self.save()
        self.serial_STOP()
        self.ser.close()

    def save(self, timestamp=True):
        # Save data
        self.dab.save_to_file(directory=self.directory, name='dab_' + self.name, comment=self.comment,
                              timestamp=timestamp)
        self.dab.pprint_to_file(os.path.join(self.directory, 'dab_' + self.name + '.txt'))

    def iterate_dab_measure(self, mod='sps'):

        self.log.info("Start iterating the DAB and measuring: " + mod)

        # Default measurement result
        for k in l_meas_keys:
            self.dab[meas_prefix + mod + '_' + k] = np.full_like(self.dab.mesh_V1, np.nan)

        try:
            # Iterate the V1,V2,P meshgrid
            for vec_vvp in np.ndindex(self.dab.mesh_V1.shape):
                # db.debug(vec_vvp,
                #          self.dab.mesh_P[vec_vvp].item(),
                #          self.dab.mesh_V1[vec_vvp].item(),
                #          self.dab.mesh_V2[vec_vvp].item(),
                #          self.dab['mod_' + mod + '_phi'][vec_vvp].item(),
                #          self.dab['mod_' + mod + '_tau1'][vec_vvp].item(),
                #          self.dab['mod_' + mod + '_tau2'][vec_vvp].item())
                V1 = self.dab.mesh_V1[vec_vvp].item()
                V2 = self.dab.mesh_V2[vec_vvp].item()
                P = self.dab.mesh_P[vec_vvp].item()
                phi = int(self.dab['mod_' + mod + '_phi'][vec_vvp].item() / (2 * np.pi) * 10000)
                tau1 = int(self.dab['mod_' + mod + '_tau1'][vec_vvp].item() / (2 * np.pi) * 10000)
                tau2 = int(self.dab['mod_' + mod + '_tau2'][vec_vvp].item() / (2 * np.pi) * 10000)
                dt1 = int(self.dab.t_dead1)
                dt2 = int(self.dab.t_dead2)
                # Only run datapoint if all params are valid
                if not np.any(np.isnan(list([phi, tau1, tau2]))):
                    self.log.info('Set voltages to: V1 = {}V  V2 = {}V  Expected power: {}W'.format(V1, V2, P))
                    apply = input('Apply next data point? ({}, {}, {})'.format(phi, tau1, tau2)
                                  + '\n[<ENTER>=Yes, n=Skip, p=Pause, s=STOP]\n> ')
                    if 's' in apply:
                        self.serial_STOP()
                        break
                    if 'p' in apply:
                        self.serial_STOP()
                        resume = input('Resume with current data point?'
                                       + '\n[<ENTER>=Yes, s=STOP]\n> ')
                        if 's' in resume:
                            break
                    if 'n' in apply:
                        continue
                    self.serial_set(phi, tau1, tau2, dt1, dt2)
                    # Now read the measuring results
                    self.log.info('Please input measurement in following order:\n'
                                  + ' '.join(map(str, l_meas_keys)))
                    meas = input('[.=decimal, <space>=separator, nan=NaN, <empty>,s=STOP]\n> ')
                    if not meas or 's' in meas:
                        self.serial_STOP()
                        break
                    try:
                        # Convert values to float and put them in an array
                        meas_a = np.float_(meas.split())
                        meas_d = dict(zip(l_meas_keys, meas_a.tolist()))
                        # Save meas to dab
                        for k in l_meas_keys:
                            self.dab[meas_prefix + mod + '_' + k][vec_vvp] = meas_d[k]
                    except:
                        # Very crude second chance on typo, but better than no second chance at all ;)
                        self.log.error('Wrong input, please try again:\n'
                                       + ' '.join(map(str, l_meas_keys)))
                        meas = input('[.=decimal, <space>=separator, nan=NaN]\n> ')
                        # Convert values to float and put them in an array
                        meas_a = np.float_(meas.split())
                        meas_d = dict(zip(l_meas_keys, meas_a.tolist()))
                        # Save meas to dab
                        for k in l_meas_keys:
                            self.dab[meas_prefix + mod + '_' + k][vec_vvp] = meas_d[k]
                    self.serial_set(0, 5000, 5000, dt1, dt2)
        except KeyboardInterrupt:
            # Emergency stop in case we hit <Ctrl+c> and save what we have
            self.serial_STOP()
            self.save()
            self.log.error('You aborted the measurement. DAB Stopped. Data saved.')
        finally:
            self.serial_STOP()
            self.save()
            self.log.info('DAB Stopped. Data saved.')

        # Iteration is complete
        self.log.info('Measurement complete for modulation: ' + mod)

    def serial_set(self, phi, tau1, tau2, dt1, dt2):
        # state True indicates everything is fine
        state = True

        # Send one data packet, and then we must receive one data packet and a byte string
        dab_params = struct.pack('>HhHHHH', self.STATE_RUN, int(phi), int(tau1), int(tau2), int(dt1), int(dt2))
        len = self.ser.write(dab_params)
        # We have to read this to clear the buffer
        dab_params_rx = self.ser.read(len)
        # We have to read this to clear the buffer
        msg = self.ser.readline()

        if (dab_params_rx == dab_params):
            self.log.debug('DAB Params send successfully: ' + str(msg))
            state = True
        else:
            self.log.error('DAB Params not send successfully! Aborting!')
            # self.serial_STOP()
            state = False

        return state

    def serial_STOP(self):
        self.log.error('DAB Stopped!')
        # Send one data packet, and then we must receive one data packet and a byte string
        dab_params = struct.pack('>HhHHHH', self.STATE_STOP, 0, 5000, 5000, 50, 50)
        len = self.ser.write(dab_params)
        # We have to read this to clear the buffer
        self.ser.read(len)
        # We have to read this to clear the buffer
        self.ser.readline()


def main_init():
    parser = argparse.ArgumentParser()
    # parser.add_argument("configfile", help="config file")
    parser.add_argument("-l", help="Log to file: <datetime>_<name>.log with -l <name>")
    parser.add_argument("-d", help="Set log output to debug level", action="store_true")
    args = parser.parse_args()

    # if os.path.isfile(args.configfile):
    #     config = yaml.load(open(args.configfile))
    # else:
    #     logging.error("[ERROR] configfile '{}' does not exist!".format(args.configfile), file=sys.stderr)
    #     sys.exit(1)


def plot_meas(dab, name, comment, directory, mod_keys):
    ## Plotting
    info("\nStart Plotting\n")
    debug(mod_keys)

    # When dim=1 the v1_middle calc does not work.
    # Therefore, we stretch the array to use the same algo for every data.
    # FIXME test this copy if it overwrites the original
    dab = copy.deepcopy(dab)
    if dab.mesh_V1.shape[1] == 1:
        ## Broadcast arrays for plotting
        for k, v in dab.items():
            if k.startswith(('mesh_', 'mod_', 'sim_', 'meas_')):
                dab[k] = np.broadcast_to(dab[k], (dab.V2_step, 3, dab.P_step))

    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab.mesh_P)[1] / 2)
    info('View plane: U_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB(latex=True)

    for m in mod_keys:
        # i_l and v_ds plots
        plt.new_fig(nrows=1, ncols=3, tab_title=m + ' i_rms ZVS')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_I_HF1'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_2 \:/\: \mathrm{V}$',
                             title=r'$I_{1} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_v_ds_S11_sw_on'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$u_\mathrm{DS,S11,sw-on} \:/\: \mathrm{V}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_v_ds_S23_sw_on'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$u_\mathrm{DS,S23,sw-on} \:/\: \mathrm{V}$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig2'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # Total loss
        plt.new_fig(nrows=1, ncols=3, tab_title=m + ' Total Loss')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_P_dc1'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_2 \:/\: \mathrm{V}$',
                             title=r'$P_\mathrm{in,Meas} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_P_dc2'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{out,Meas} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_P_dc2'][:, v1_middle, :]
                             / dab['meas_' + m + '_P_dc1'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$\eta$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig3'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

    info('Plotting is done!')
    # Finally show everything
    plt.show()


def plot_meas_ib(dab, name, comment, directory, mod_keys):
    ## Plotting
    info("\nStart Plotting\n")
    debug(mod_keys)

    # When dim=1 the v1_middle calc does not work.
    # Therefore, we stretch the array to use the same algo for every data.
    # FIXME test this copy if it overwrites the original
    dab = copy.deepcopy(dab)
    if dab.mesh_V1.shape[1] == 1:
        ## Broadcast arrays for plotting
        for k, v in dab.items():
            if k.startswith(('mesh_', 'mod_', 'sim_', 'meas_')):
                dab[k] = np.broadcast_to(dab[k], (dab.V2_step, 3, dab.P_step))

    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab.mesh_P)[1] / 2)
    info('View plane: U_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB(latex=True)

    for m in mod_keys:
        # i_l and v_ds plots
        plt.new_fig(nrows=1, ncols=2, tab_title=m + ' i_rms ZVS')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_I_HF1'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_2 \:/\: \mathrm{V}$',
                             title=r'$I_{1} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_i_HF1_S11_sw_on'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$i_\mathrm{1,S11,sw-on} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_v_ds_S11_sw_on'][:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$u_\mathrm{DS,S11,sw-on} \:/\: \mathrm{V}$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig1'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # Total loss
        plt.new_fig(nrows=1, ncols=3, tab_title=m + ' Total Loss')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_I_dc1'][:, v1_middle, :] * dab.mesh_V1[:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_2 \:/\: \mathrm{V}$',
                             title=r'$P_\mathrm{in,Meas} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_I_dc2'][:, v1_middle, :] * dab.mesh_V2[:, v1_middle, :],
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{out,Meas} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['meas_' + m + '_I_dc2'][:, v1_middle, :] * dab.mesh_V2[:, v1_middle, :]
                             / (dab['meas_' + m + '_I_dc1'][:, v1_middle, :] * dab.mesh_V1[:, v1_middle, :]),
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$\eta$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig3'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

    info('Plotting is done!')
    # Finally show everything
    plt.show()


def load_dab():
    # Select File Folder and Path
    dir = 'name-of-npz-file'
    path = '../dab_modulation_output/'

    # Extract name
    name = os.path.splitext(dir.split('_', 2)[2])[0]
    # Loading
    dab_file = os.path.join(path, dir, name + '.npz')
    dab_file = os.path.expanduser(dab_file)
    dab_file = os.path.expandvars(dab_file)
    dab_file = os.path.abspath(dab_file)
    dab = ds.load_from_file(dab_file)
    # dab.pprint()
    db.debug(dab._comment)


def serial_control_manual():
    state = 0xAA
    # Some mod mcl data point for test:
    # phi = 1035
    # tau1 = 3065
    # tau2 = 4100
    # phi_l = [787, 1041, 1245, 692, 894, 1124, 1402, 0]
    # tau1_l = [2331, 3083, 3685, 5000, 5000, 5000, 5000, 5000]
    # tau2_l = [3118, 4124, 4930, 5000, 5000, 5000, 5000, 5000]
    phi_l = [787, 1047, 1124, 0]
    tau1_l = [2331, 3101, 5000, 5000]
    tau2_l = [3118, 4148, 5000, 5000]
    dt1 = 50
    dt2 = 50

    # Serial Port Settings
    # port = '/dev/ttyACM0'
    port = '/dev/ttyUSB0'
    baud = 115200

    ser = serial.Serial(port, baud, timeout=1)

    for phi, tau1, tau2 in zip(phi_l, tau1_l, tau2_l):
        input('Apply next data point? ({}, {}, {})'.format(phi, tau1, tau2))
        dab_params = struct.pack('>HhHHHH', state, phi, tau1, tau2, dt1, dt2)
        len = ser.write(dab_params)
        dab_params_rx = ser.read(len)
        msg = ser.readline()
        db.debug(msg)

        if (dab_params_rx == dab_params and 'OK' in str(msg)):
            db.info('DAB Params send successfully')

    ser.close()


def trial_serial():
    state = 0xAA
    # Some mod mcl data point for test:
    phi = 1035
    tau1 = 3065
    tau2 = 4100
    # Fake data:
    phi = 2500
    tau1 = 5000
    tau2 = 2500
    dt1 = 50
    dt2 = 50

    # Serial Port Settings
    # port = '/dev/ttyACM0'
    port = '/dev/ttyUSB0'
    baud = 115200

    ser = serial.Serial(port, baud, timeout=1)

    dab_params = struct.pack('>HhHHHH', state, phi, tau1, tau2, dt1, dt2)
    # < b'\xaa\x00\x0b\x04\xf9\x0b\x04\x102\x002\x00'
    # > b'\x00\xaa\x04\x0b\x0b\xf9\x10\x04\x002\x002'
    # Test
    # b'\x00\xaa\t\xc4\x13\x88\t\xc4\x002\x002'
    # 12
    # b'\x00\xaa\t\xc4\x13\x88\t\xc4\x002\x002'
    # b'OK: RX Data Packet\n'
    db.debug(dab_params)
    len = ser.write(dab_params)
    db.debug(len)
    dab_params_rx = ser.read(len)
    db.debug(dab_params_rx)
    msg = ser.readline()
    db.debug(msg)

    if (dab_params_rx == dab_params):
        db.info('DAB Params send successfully')

    ser.close()


def _main_dummy():
    return


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of DAB Optimizer ...")

    # Set some defaults
    # port = '/dev/ttyACM0'
    port = '/dev/ttyUSB0'
    directory = '../dab_modulation_output/'
    filename = '../dab_modulation_output/dab_sim_Gv5_Ls_85uH__Lc1_25620uH__Lc2_611uH_v1-v7-p7.npz'

    # Do some basic init like logging, args, etc.
    # main_init()

    control = DAB_Control(filename, directory, port)
    control.iterate_dab_measure('sps')
    # control.iterate_dab_measure('mcl')
    # control.iterate_dab_measure('zvs')
    control.save(timestamp=False)
    # plot_meas(control.dab, control.name, control.comment, control.directory, ['sps', 'mcl', 'zvs'])
    # plot_meas(control.dab, control.name, control.comment, control.directory, ['mcl'])
    plot_meas_ib(control.dab, control.name, control.comment, control.directory, ['sps'])

    # # Only load and plot
    # measfile = '../dab_modulation_output/lab_measurement__dab_sim_Gv5_Ls_85uH__Lc1_25620uH__Lc2_611uH_v1-v7-p7.npz/dab_lab_measurement.npz'
    # dab = ds.load_from_file(measfile)
    # plot_meas(dab, 'lab_meas_replot', 'Lab meas replot.', os.path.dirname(measfile), ['sps'])

    # Load and then do something
    # load_dab()
    # serial_control_manual()

    # Serial test
    # trial_serial()
