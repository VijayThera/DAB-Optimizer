#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

import os
import sys
import serial
import struct

import numpy as np
import math
# from datetime import datetime
# import logging
import argparse

import dab_datasets as ds
from debug_tools import *


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

    # Test the logging
    # info("test")
    # debug("test")
    # warning("test")
    # error("test")

    # Logging with logger is problematic!
    # print("this should be before logger init")
    # # Set up the logger
    # if args.d or db.DEBUG or __debug__:
    #     loglevel = logging.DEBUG
    # else:
    #     loglevel = logging.INFO
    #
    # format = '%(asctime)s %(module)s %(levelname)s: %(message)s'
    # if args.l:
    #     logging.basicConfig(format=format, level=loglevel,
    #                         filename=str(datetime.now().strftime("%Y-%m-%d_%H%M%S")) + "_" + args.l + ".log",
    #                         encoding='utf-8', force=True)
    # else:
    #     logging.basicConfig(format=format, level=loglevel, force=True)
    #     #logging.basicConfig(format='%(asctime)s %(message)s', level=loglevel)
    #
    # # create log
    # #logging.root.setLevel(loglevel)
    # log = logging.getLogger(__name__)
    # #log.setLevel(logging.DEBUG)
    # # create console handler and set level to debug
    # # ch = logging.StreamHandler()
    # # ch.setLevel(logging.DEBUG)
    # # # create formatter
    # # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # # # add formatter to ch
    # # ch.setFormatter(formatter)
    # # # add ch to log
    # # log.addHandler(ch)
    #
    # # 'application' code
    # log.debug('debug message')
    # log.info('info message')
    # log.warning('warn message')
    # log.error('error message')
    # log.critical('critical message')
    #
    # log.debug("TEST")
    # log.debug("INT %s", 5)
    # log.info("INFO")
    # d = {"name" : "John", "age": 10}
    # log.info('Test dict: %s', d)
    #
    # print("this should be after logger init")
    # sys.exit(0)


def load_dab():
    # Select File Folder and Path
    dir = '2023-04-28_01:34:11_mod_sps_mcl_v1-v7-p7'
    path = ''

    # Extract name
    name = os.path.splitext(dir.split('_', 2)[2])[0]
    # Loading
    dab_file = os.path.join(path, dir, name + '.npz')
    dab_file = os.path.expanduser(dab_file)
    dab_file = os.path.expandvars(dab_file)
    dab_file = os.path.abspath(dab_file)
    dab = ds.load_from_file(dab_file)
    # dab.pprint()
    debug(dab._comment)


def serial_control_manual():
    state = 0xAA
    # Some mod mcl data point for test:
    # phi = 1035
    # tau1 = 3065
    # tau2 = 4100
    phi_l = [787, 1041, 1245, 692, 894, 1124, 1402, 0]
    tau1_l = [2331, 3083, 3685, 5000, 5000, 5000, 5000, 5000]
    tau2_l = [3118, 4124, 4930, 5000, 5000, 5000, 5000, 5000]
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
        debug(msg)

        if (dab_params_rx == dab_params):
            info('DAB Params send successfully')


def trail_serial():
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
    debug(dab_params)
    len = ser.write(dab_params)
    debug(len)
    dab_params_rx = ser.read(len)
    debug(dab_params_rx)
    msg = ser.readline()
    debug(msg)

    if (dab_params_rx == dab_params):
        info('DAB Params send successfully')


def _main_dummy():
    return


# ---------- MAIN ----------
if __name__ == '__main__':
    info("Start of DAB Optimizer ...")
    # Do some basic init like logging, args, etc.
    main_init()

    # Load and then do something
    load_dab()
    serial_control_manual()

    # Serial test
    # trail_serial()
