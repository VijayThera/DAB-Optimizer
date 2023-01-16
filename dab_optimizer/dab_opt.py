#!/usr/bin/python3
# coding: utf-8
# python >= 3.10

import os
import sys

import numpy as np
from datetime import datetime
import logging
import argparse

import classes_datasets as ds
from debug_tools import *
import mod_cpm
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
        # "dab_results_" + used module (e.g. "mod_cpm_") + value name (e.g. "phi")
        dab_results_mod_cpm_phi: mod_cpm calculated values for phi
        dab_results_mod_cpm_tau1: mod_cpm calculated values for tau1
        dab_results_mod_cpm_tau2: mod_cpm calculated values for tau1
        dab_results_sim_cpm_iLs: simulation results with mod_cpm for iLs
        dab_results_sim_cpm_S11_p_sw:

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
def test_dab():
    """
    Run the complete optimization procedure
    """
    # Set the basic DAB Specification
    # Setting it this way enables tab completion.
    dab_specs = ds.DAB_Specification()
    dab_specs.V1_nom = 700
    dab_specs.V1_min = 600
    dab_specs.V1_max = 800
    dab_specs.V1_step = 3
    dab_specs.V2_nom = 235
    dab_specs.V2_min = 175
    dab_specs.V2_max = 295
    dab_specs.V2_step = 3
    dab_specs.P_min = 400
    dab_specs.P_max = 2200
    dab_specs.P_nom = 2000
    dab_specs.P_step = 3
    dab_specs.n = 2.99
    dab_specs.L_s = 84e-6
    dab_specs.L_m = 599e-6
    dab_specs.fs_nom = 200000

    # Object to store all generated data
    dab_results = ds.DAB_Results()
    # gen mesh manually
    # TODO provide a generator function for this in the DAB_Results class
    # dab_results.mesh_V1, dab_results.mesh_V2, dab_results.mesh_P = np.meshgrid(
    #     np.linspace(dab_specs.V1_min, dab_specs.V1_max, int(dab_specs.V1_step)),
    #     np.linspace(dab_specs.V2_min, dab_specs.V2_max, int(dab_specs.V2_step)),
    #     np.linspace(dab_specs.P_min, dab_specs.P_max, int(dab_specs.P_step)), sparse=False)
    dab_results.gen_meshes(
        dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step,
        dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step,
        dab_specs.P_min, dab_specs.P_max, dab_specs.P_step)

    # Modulation Calculation
    # TODO how to name the arrays according to some kind of given pattern?
    dab_results.mod_phi, dab_results.mod_tau1, dab_results.mod_tau2 = mod_cpm.calc_modulation(dab_specs.n,
                                                                                              dab_specs.L_s,
                                                                                              dab_specs.fs_nom,
                                                                                              dab_results.mesh_V1,
                                                                                              dab_results.mesh_V2,
                                                                                              dab_results.mesh_P)

    # Simulation
    dab_results.sim_iLs, dab_results.sim_S11_p_sw = sim_gecko.start_sim(dab_results.mesh_V1,
                                                                        dab_results.mesh_V2,
                                                                        dab_results.mesh_P,
                                                                        dab_results.mod_phi,
                                                                        dab_results.mod_tau1,
                                                                        dab_results.mod_tau2)
    debug("sim_iLs: \n", dab_results.sim_iLs)
    debug("sim_S11_p_sw: \n", dab_results.sim_S11_p_sw)

    # Plotting
    pw = plotWindow()
    fig = plot_dab.plot_modulation(dab_results.mesh_V2,
                                   dab_results.mesh_P,
                                   dab_results.mod_phi,
                                   dab_results.mod_tau1,
                                   dab_results.mod_tau2)
    pw.addPlot("DAB Modulation Angles", fig)
    fig = plot_dab.plot_rms_current(dab_results.mesh_V2,
                                    dab_results.mesh_P,
                                    dab_results.sim_iLs)
    pw.addPlot("iLs", fig)
    fig = plot_dab.plot_rms_current(dab_results.mesh_V2,
                                    dab_results.mesh_P,
                                    dab_results.sim_S11_p_sw)
    pw.addPlot("S11 p_sw", fig)
    # plot_dab.show_plot()
    pw.show()

    # Saving
    # save_to_file(dab_specs, dab_results, name='test-save', comment='This is a saving test with random data!')
    # save_to_file(dab_specs, dab_results, name='test-save', timestamp=False, comment='This is a saving test with random data!')

    # Loading
    # dab_specs_loaded, dab_results_loaded = load_from_file('test-save.npz')
    # dab_specs_loaded.pprint()
    # dab_results_loaded.pprint()

    # add some false data, should output an error log or warning
    # dab_results_loaded.foo = np.array([1, 2, 3])
    # dab_results_loaded.bar = "test"

    # Test the logging
    # info("test")
    # debug("test")
    # warning("test")
    # error("test")

def test_plot():
    # Loading
    dab_specs, dab_results = load_from_file('test-save.npz')
    # dab_specs.pprint()
    # dab_results.pprint()

    info("\nStart Plotting\n")
    plt_dab = plot_dab.Plot_DAB()
    plt_dab.new_fig(1, 3)
    plt_dab.plot_modulation(plt_dab.figs_axes[-1],
                            dab_results.mesh_P[:, 1, :],
                            dab_results.mesh_V2[:, 1, :],
                            dab_results.mod_phi[:, 1, :],
                            dab_results.mod_tau1[:, 1, :],
                            dab_results.mod_tau2[:, 1, :])

    plt_dab.new_fig(1, 3)
    plt_dab.plot_modulation(plt_dab.figs_axes[-1],
                            dab_results.mesh_P[:, 1, :],
                            dab_results.mesh_V2[:, 1, :],
                            dab_results.mod_phi[:, 1, :],
                            dab_results.mod_tau1[:, 1, :],
                            dab_results.mod_tau2[:, 1, :])

    # now redraw the first fig
    plt_dab.plot_modulation(plt_dab.figs_axes[0],
                            dab_results.mesh_P[:, 1, :],
                            dab_results.mesh_V2[:, 1, :],
                            dab_results.mod_phi[:, 1, :],
                            dab_results.mod_phi[:, 1, :],
                            dab_results.mod_tau2[:, 1, :])

    plt_dab.show()


# ---------- MAIN ----------
if __name__ == '__main__':
    info("Start of DAB Optimizer ...")
    # Do some basic init like logging, args, etc.
    main_init()
    # Test the DAB functions
    #test_dab()
    # Test the Plot functions
    test_plot()

    # sys.exit(0)
