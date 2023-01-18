#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

import os
import sys

import numpy as np
import math
from datetime import datetime
import logging
import argparse

import dab_datasets as ds
from debug_tools import *
import mod_sps
import sim_gecko
import plot_dab

from plotWindow import plotWindow


def save_to_file(dab_specs: ds.DAB_Specification, dab_results: ds.DAB_Results,
                 directory=str(), name=str(), timestamp=True, comment=str()):
    """
    Save everything (except plots) in one file.
    WARNING: Existing files will be overwritten!

    File is ZIP compressed and contains several named np.ndarray objects:
        # Starting with "dab_specs_" are for the DAB_Specification
        dab_specs_keys: containing the dict keys as strings
        dab_specs_values: containing the dict values as float
        # Starting with "dab_results_" are for the DAB_Results
        # TODO shoud I store generated meshes? Maybe not but regenerate them after loading a file.
        dab_results_mesh_V1: generated mesh
        dab_results_mesh_V2: generated mesh
        dab_results_mesh_P: generated mesh
        # String is constructed as follows:
        # "dab_results_" + used module (e.g. "mod_sps_") + value name (e.g. "phi")
        dab_results_mod_sps_phi: mod_sps calculated values for phi
        dab_results_mod_sps_tau1: mod_sps calculated values for tau1
        dab_results_mod_sps_tau2: mod_sps calculated values for tau1
        dab_results_sim_sps_iLs: simulation results with mod_sps for iLs
        dab_results_sim_sps_S11_p_sw:

    :param comment:
    :param dab_specs:
    :param dab_results:
    :param directory: Folder where to save the files
    :param name: String added to the filename. Without file extension. Datetime may prepend the final name.
    :param timestamp: If the datetime should prepend the final name. default True
    """
    # Temporary Dict to hold the name/array (key/value) pairs to be stored by np.savez
    # Arrays to save to the file. Each array will be saved to the output file with its corresponding keyword name.
    kwds = dict()
    kwds['dab_specs_keys'], kwds['dab_specs_values'] = dab_specs.export_to_array()

    for k, v in dab_results.items():
        # TODO filter for mesh here
        kwds['dab_results_' + k] = v

    # Add some descriptive data to the file
    # Adding a timestamp, it may be useful
    kwds['_timestamp'] = np.asarray(datetime.now().isoformat())
    # Adding a comment to the file, hopefully a descriptive one
    if comment:
        kwds['_comment'] = np.asarray(comment)

    # Adding a timestamp to the filename if requested
    if timestamp:
        if name:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
        else:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        if name:
            filename = name
        else:
            # set some default non-empty filename
            filename = "dab_opt_dataset"

    if (not directory) or os.path.isdir(directory):
        file = os.path.join(directory, filename)
    else:
        warning("Directory does not exist!")
        return

    # numpy saves everything for us in a handy zip file
    # np.savez(file=file, **kwds)
    np.savez_compressed(file=file, **kwds)


def load_from_file(file: str) -> tuple[ds.DAB_Specification, ds.DAB_Results]:
    """
    Load everything from the given .npz file.
    :param file: a .nps filename or file-like object, string, or pathlib.Path
    :return: two objects with type DAB_Specification and DAB_Results
    """
    dab_specs = ds.DAB_Specification()
    dab_results = ds.DAB_Results()
    # Open the file and parse the data
    with np.load(file) as data:
        spec_keys = None
        spec_values = None
        for k, v in data.items():
            if k.startswith('dab_results_'):
                dab_results[k.removeprefix('dab_results_')] = v
            if str(k) == 'dab_specs_keys':
                spec_keys = v
            if str(k) == 'dab_specs_values':
                spec_values = v
            if str(k) == '_timestamp':
                dab_results[k] = v
            if str(k) == '_comment':
                dab_results[k] = v
        # We can not be sure if specs where in the file but hopefully there was
        if not (spec_keys is None and spec_values is None):
            dab_specs.import_from_array(spec_keys, spec_values)
        # TODO regenerate meshes here
    return dab_specs, dab_results


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


@timeit
def dab_sim_save():
    """
    Run the complete optimization procedure and save the results in a file
    """
    # Set the basic DAB Specification
    Dab_Specs = ds.DAB_Specification()
    Dab_Specs.V1_nom = 700
    Dab_Specs.V1_min = 600
    Dab_Specs.V1_max = 800
    # Dab_Specs.V1_step = math.floor((Dab_Specs.V1_max - Dab_Specs.V1_min) / 10 + 1)
    Dab_Specs.V1_step = 3
    Dab_Specs.V2_nom = 235
    Dab_Specs.V2_min = 175
    Dab_Specs.V2_max = 295
    # Dab_Specs.V2_step = math.floor((Dab_Specs.V2_max - Dab_Specs.V2_min) / 5 + 1)
    Dab_Specs.V2_step = 4
    Dab_Specs.P_min = 400
    Dab_Specs.P_max = 2200
    Dab_Specs.P_nom = 2000
    # Dab_Specs.P_step = math.floor((Dab_Specs.P_max - Dab_Specs.P_min) / 100 + 1)
    Dab_Specs.P_step = 5
    Dab_Specs.n = 2.99
    Dab_Specs.L_s = 84e-6
    Dab_Specs.L_m = 599e-6
    Dab_Specs.fs_nom = 200000

    # Object to store all generated data
    Dab_Results = ds.DAB_Results()
    # Generate meshes
    Dab_Results.gen_meshes(
        Dab_Specs.V1_min, Dab_Specs.V1_max, Dab_Specs.V1_step,
        Dab_Specs.V2_min, Dab_Specs.V2_max, Dab_Specs.V2_step,
        Dab_Specs.P_min, Dab_Specs.P_max, Dab_Specs.P_step)

    # Modulation Calculation
    Dab_Results.mod_sps_phi, \
        Dab_Results.mod_sps_tau1, \
        Dab_Results.mod_sps_tau2 = mod_sps.calc_modulation(Dab_Specs.n,
                                                           Dab_Specs.L_s,
                                                           Dab_Specs.fs_nom,
                                                           Dab_Results.mesh_V1,
                                                           Dab_Results.mesh_V2,
                                                           Dab_Results.mesh_P)

    # TODO where to save??? spec only float...
    simfilepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC.ipes'
    timestep = 100e-12
    simtime = 15e-6

    # Simulation
    # d_sim = sim_gecko.start_sim(Dab_Results.mesh_V1,
    #                             Dab_Results.mesh_V2,
    #                             Dab_Results.mod_sps_phi,
    #                             Dab_Results.mod_sps_tau1,
    #                             Dab_Results.mod_sps_tau2,
    #                             simfilepath, timestep, simtime)

    # Simulation
    Dab_Sim = sim_gecko.Sim_Gecko()
    # d_sim = Dab_Sim.start_sim_threads(Dab_Results.mesh_V1,
    #                                   Dab_Results.mesh_V2,
    #                                   Dab_Results.mod_sps_phi,
    #                                   Dab_Results.mod_sps_tau1,
    #                                   Dab_Results.mod_sps_tau2,
    #                                   simfilepath, timestep, simtime)

    d_sim = Dab_Sim.start_sim_multi(Dab_Results.mesh_V1,
                                Dab_Results.mesh_V2,
                                Dab_Results.mod_sps_phi,
                                Dab_Results.mod_sps_tau1,
                                Dab_Results.mod_sps_tau2,
                                simfilepath, timestep, simtime)

    # Unpack the results
    for k, v in d_sim.items():
        Dab_Results['sim_' + k] = v

    debug("sim_i_Ls: \n", Dab_Results.sim_i_Ls)
    debug("sim_S11_p_sw: \n", Dab_Results.sim_S11_p_sw)

    # Plotting
    info("\nStart Plotting\n")
    Plot_Dab = plot_dab.Plot_DAB()
    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='SPS Overview')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.mod_sps_phi[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][0],
                             xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_i_Ls[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][1],
                             xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_S11_p_sw[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][2],
                             xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='SPS Power')
    Plot_Dab.plot_3by1(Plot_Dab.figs_axes[-1],
                      Dab_Results.mesh_P[:, 1, :],
                      Dab_Results.mesh_V2[:, 1, :],
                      Dab_Results.sim_p_dc1[:, 1, :],
                      Dab_Results.sim_S11_p_sw[:, 1, :],
                      Dab_Results.sim_S11_p_cond[:, 1, :],
                      'P / W',
                      'U2 / V',
                      'p_dc1',
                      'S11_p_sw',
                      'S11_p_cond')

    Plot_Dab.show()

    # Calc power deviation from expected power target
    # power_deviation = mesh_P[vec_vvp].item() and values_mean['mean']['p_dc1'] / mesh_P[vec_vvp].item()
    # debug("power_sim: %f / power_target: %f -> power_deviation: %f" % (values_mean['mean']['p_dc1'], mesh_P[vec_vvp].item(), power_deviation))

    # Saving
    # save_to_file(Dab_Specs, Dab_Results, name='mod_sps_sim_v21-v25-p19',
    #              comment='Simulation results for mod_sps with V1 10V res, V2 5V res and P 100W res.')


@timeit
def test_dab():
    """
    Run the complete optimization procedure
    """
    # Set the basic DAB Specification
    # Setting it this way enables tab completion.
    Dab_Specs = ds.DAB_Specification()
    Dab_Specs.V1_nom = 700
    Dab_Specs.V1_min = 600
    Dab_Specs.V1_max = 800
    Dab_Specs.V1_step = 2
    Dab_Specs.V2_nom = 235
    Dab_Specs.V2_min = 175
    Dab_Specs.V2_max = 295
    Dab_Specs.V2_step = 3
    Dab_Specs.P_min = 400
    Dab_Specs.P_max = 2200
    Dab_Specs.P_nom = 2000
    Dab_Specs.P_step = 3
    Dab_Specs.n = 2.99
    Dab_Specs.L_s = 84e-6
    Dab_Specs.L_m = 599e-6
    Dab_Specs.fs_nom = 200000

    # Object to store all generated data
    Dab_Results = ds.DAB_Results()
    # gen mesh manually
    # TODO provide a generator function for this in the DAB_Results class
    # Dab_Results.mesh_V1, Dab_Results.mesh_V2, Dab_Results.mesh_P = np.meshgrid(
    #     np.linspace(Dab_Specs.V1_min, Dab_Specs.V1_max, int(Dab_Specs.V1_step)),
    #     np.linspace(Dab_Specs.V2_min, Dab_Specs.V2_max, int(Dab_Specs.V2_step)),
    #     np.linspace(Dab_Specs.P_min, Dab_Specs.P_max, int(Dab_Specs.P_step)), sparse=False)
    Dab_Results.gen_meshes(
        Dab_Specs.V1_min, Dab_Specs.V1_max, Dab_Specs.V1_step,
        Dab_Specs.V2_min, Dab_Specs.V2_max, Dab_Specs.V2_step,
        Dab_Specs.P_min, Dab_Specs.P_max, Dab_Specs.P_step)

    # Modulation Calculation
    # TODO how to name the arrays according to some kind of given pattern?
    # SPS Modulation
    Dab_Results.mod_sps_phi, Dab_Results.mod_sps_tau1, Dab_Results.mod_sps_tau2 = mod_sps.calc_modulation(Dab_Specs.n,
                                                                                                          Dab_Specs.L_s,
                                                                                                          Dab_Specs.fs_nom,
                                                                                                          Dab_Results.mesh_V1,
                                                                                                          Dab_Results.mesh_V2,
                                                                                                          Dab_Results.mesh_P)

    # RMS Modulation
    # Dab_Results.mod_rms_phi, Dab_Results.mod_rms_tau1, Dab_Results.mod_rms_tau2 = mod_rms.calc_modulation(Dab_Specs.n,
    #                                                                                                       Dab_Specs.L_s,
    #                                                                                                       Dab_Specs.fs_nom,
    #                                                                                                       Dab_Results.mesh_V1,
    #                                                                                                       Dab_Results.mesh_V2,
    #                                                                                                       Dab_Results.mesh_P)


    # TODO where to save??? spec only float...
    simfilepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC.ipes'
    timestep = 100e-12
    simtime = 15e-6

    # Simulation
    d_sim = sim_gecko.start_sim(Dab_Results.mesh_V1,
                                Dab_Results.mesh_V2,
                                Dab_Results.mod_sps_phi,
                                Dab_Results.mod_sps_tau1,
                                Dab_Results.mod_sps_tau2,
                                simfilepath, timestep, simtime)

    # Unpack the results
    # TODO maybe put this as a function in DAB_Results
    for k, v in d_sim.items():
        Dab_Results['sim_' + k] = v

    debug("sim_i_Ls: \n", Dab_Results.sim_i_Ls)
    debug("sim_S11_p_sw: \n", Dab_Results.sim_S11_p_sw)

    # Plotting
    info("\nStart Plotting\n")
    Plot_Dab = plot_dab.Plot_DAB()
    # Mod SPS
    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='SPS Overview')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.mod_sps_phi[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][0],
                             xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_i_Ls[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][1],
                             xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_S11_p_sw[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][2],
                             xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='SPS Power')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_p_dc1[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][0],
                             xlabel='P / W', ylabel='U2 / V', title='p_dc1 / W')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_S11_p_sw[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][1],
                             xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_S11_p_cond[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][2],
                             xlabel='P / W', ylabel='U2 / V', title='S11_p_cond / W')

    # Mod MCL
    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='MCL Modulation')
    Plot_Dab.plot_modulation(Plot_Dab.figs_axes[-1],
                      Dab_Results.mesh_P[:, 1, :],
                      Dab_Results.mesh_V2[:, 1, :],
                      Dab_Results.sim_p_dc1[:, 1, :],
                      Dab_Results.sim_S11_p_sw[:, 1, :],
                      Dab_Results.sim_S11_p_cond[:, 1, :])

    Plot_Dab.show()

    # Saving
    save_to_file(Dab_Specs, Dab_Results, directory='/mnt/MA LEA/LEA/Workdir/dab_optimizer_output', name='test-save', comment='This is a saving test with random data!')
    # save_to_file(Dab_Specs, Dab_Results, name='test-save', timestamp=False, comment='This is a saving test with random data!')

    # Loading
    # dab_specs_loaded, dab_results_loaded = load_from_file('test-save.npz')
    # dab_specs_loaded.pprint()
    # dab_results_loaded.pprint()

    # add some false data, should output an error log or warning
    # dab_results_loaded.foo = np.array([1, 2, 3])
    # dab_results_loaded.bar = "test"


def test_plot():
    # Loading
    Dab_Specs, Dab_Results = load_from_file('/mnt/MA LEA/LEA/Workdir/dab_optimizer_output/test-sps-save.npz')
    Dab_Specs.pprint()
    # Dab_Results.pprint()

    info("\nStart Plotting\n")
    Plot_Dab = plot_dab.Plot_DAB()
    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='SPS Overview')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.mod_sps_phi[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][0],
                             xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_i_Ls[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][1],
                             xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_S11_p_sw[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][2],
                             xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='SPS Power')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_p_dc1[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][0],
                             xlabel='P / W', ylabel='U2 / V', title='sim_p_dc1 / W')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_S11_p_sw[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][1],
                             xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')
    Plot_Dab.subplot_contourf(Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.sim_S11_p_cond[:, 1, :],
                             ax=Plot_Dab.figs_axes[-1][1][2],
                             xlabel='P / W', ylabel='U2 / V', title='S11_p_cond / W')
    # Plot_Dab.plot_3by1(Plot_Dab.figs_axes[-1],
    #                   Dab_Results.mesh_P[:, 1, :],
    #                   Dab_Results.mesh_V2[:, 1, :],
    #                   Dab_Results.sim_p_dc1[:, 1, :],
    #                   Dab_Results.sim_S11_p_sw[:, 1, :],
    #                   Dab_Results.sim_S11_p_cond[:, 1, :],
    #                   'P / W',
    #                   'U2 / V',
    #                   'p_dc1',
    #                   'S11_p_sw',
    #                   'S11_p_cond')

    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='SPS Modulation')
    Plot_Dab.plot_modulation(Plot_Dab.figs_axes[-1],
                            Dab_Results.mesh_P[:, 1, :],
                            Dab_Results.mesh_V2[:, 1, :],
                            Dab_Results.mod_sps_phi[:, 1, :],
                            Dab_Results.mod_sps_tau1[:, 1, :],
                            Dab_Results.mod_sps_tau2[:, 1, :])

    # now redraw the previous fig
    # Plot_Dab.plot_modulation(Plot_Dab.figs_axes[-1],
    #                         Dab_Results.mesh_P[:, 1, :],
    #                         Dab_Results.mesh_V2[:, 1, :],
    #                         Dab_Results.mod_sps_phi[:, 1, :],
    #                         Dab_Results.mod_sps_phi[:, 1, :],
    #                         Dab_Results.mod_sps_tau2[:, 1, :])

    Plot_Dab.show()


# ---------- MAIN ----------
if __name__ == '__main__':
    info("Start of DAB Optimizer ...")
    # Do some basic init like logging, args, etc.
    main_init()

    # Generate simulation data
    # dab_sim_save()

    # Test the DAB functions
    # test_dab()
    # Test the Plot functions
    test_plot()

    # sys.exit(0)
