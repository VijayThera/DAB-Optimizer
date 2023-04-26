#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

import os
import sys

# import sys

import numpy as np
import math
# from datetime import datetime
# import logging
import argparse

import dab_datasets as ds
from debug_tools import *
import mod_sps
import mod_mcl
import mod_zvs
import sim_gecko
import plot_dab


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

    if directory:
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            file = os.path.join(directory, filename)
        else:
            warning("Directory does not exist!")
            file = os.path.join(filename)
    else:
        file = os.path.join(filename)

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


def save_to_csv(dab_specs: ds.DAB_Specification, dab_results: ds.DAB_Results, key=str(), directory=str(), name=str(),
                timestamp=True):
    """
    Save one array with name 'key' out of dab_results to a csv file
    :param dab_specs:
    :param dab_results:
    :param key: name of the array in dab_results
    :param directory:
    :param name: filename without extension
    :param timestamp: if the filename should prepended with a timestamp
    """
    if timestamp:
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
    filename = os.path.join(directory + '/' + name + '_' + key + '.csv')

    if directory:
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            file = os.path.join(directory, filename)
        else:
            warning("Directory does not exist!")
            file = os.path.join(filename)
    else:
        file = os.path.join(filename)

    comment = key + ' with P: {}, V1: {} and V2: {} steps.'.format(
        int(dab_specs.P_step),
        int(dab_specs.V1_step),
        int(dab_specs.V2_step)
    )

    # Write the array to disk
    with open(file, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# ' + comment + '\n')
        outfile.write('# Array shape: {0}\n'.format(dab_results[key].shape))
        # x: P, y: V1, z(slices): V2
        outfile.write('# x: P ({}-{}), y: V1 ({}-{}), z(slices): V2 ({}-{})\n'.format(
            int(dab_specs.P_min),
            int(dab_specs.P_max),
            int(dab_specs.V1_min),
            int(dab_specs.V1_max),
            int(dab_specs.V2_min),
            int(dab_specs.V2_max)
        ))
        outfile.write('# z: V2 ' + np.array_str(dab_results.mesh_V2[:, 0, 0], max_line_width=10000) + '\n')
        outfile.write('# y: V1 ' + np.array_str(dab_results.mesh_V1[0, :, 0], max_line_width=10000) + '\n')
        outfile.write('# x: P ' + np.array_str(dab_results.mesh_P[0, 0, :], max_line_width=10000) + '\n')

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        i = 0
        for array_slice in dab_results[key]:
            # Writing out a break to indicate different slices...
            outfile.write('# V2 slice {}V\n'.format(
                (dab_specs.V2_min + i * (dab_specs.V2_max - dab_specs.V2_min) / (dab_specs.V2_step - 1))
            ))
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            # np.savetxt(outfile, array_slice, fmt='%-7.2f')
            np.savetxt(outfile, array_slice, delimiter=';')
            i += 1


def import_Coss(file: str()):
    """
    Import a csv file containing the Coss(Vds) capacitance from the MOSFET datasheet.
    This may be generated with: https://apps.automeris.io/wpd/

    Note we assume V_ds in Volt and C_oss in pF. If this is not the case scale your data accordingly!

    CSV File should look like this:
    # V_ds / V; C_oss / pF
    1,00; 900.00
    2,00; 800.00
    :param file: csv file path
    """
    file = os.path.expanduser(file)
    file = os.path.expandvars(file)
    file = os.path.abspath(file)

    # Conversion from decimal separator comma to point so that np can read floats
    # Be careful if your csv is actually comma separated! ;)
    def conv(x):
        return x.replace(',', '.').encode()

    # Read csv file
    csv_data = np.genfromtxt((conv(x) for x in open(file)), delimiter=';', dtype=float)

    # Maybe check if data is monotonically
    # Check if voltage is monotonically rising
    if not np.all(csv_data[1:, 0] >= csv_data[:-1, 0], axis=0):
        warning("The voltage in csv file is not monotonically rising!")
    # Check if Coss is monotonically falling
    if not np.all(csv_data[1:, 1] <= csv_data[:-1, 1], axis=0):
        warning("The C_oss in csv file is not monotonically falling!")

    # Rescale and interpolate the csv data to have a nice 1V step size from 0V to v_max
    # A first value with zero volt will be added
    v_max = int(np.round(csv_data[-1, 0]))
    v_interp = np.arange(v_max + 1)
    coss_interp = np.interp(v_interp, csv_data[:, 0], csv_data[:, 1])
    # Since we now have a evenly spaced vector where x corespond to the element-number of the vector
    # we dont have to store x (v_interp) with it.
    # To get Coss(V) just get the array element coss_interp[V]

    # np.savetxt('coss.csv', coss_interp, delimiter=';')
    return coss_interp


@timeit
def integrate_Coss(coss):
    # Integrate from 0 to v
    def integrate(v):
        v_interp = np.arange(v + 1)
        coss_v = np.interp(v_interp, np.arange(coss.shape[0]), coss)
        return np.trapz(coss_v)

    coss_int = np.vectorize(integrate)
    # get an qoss vector that has the resolution 1V from 0 to V_max
    v_vec = np.arange(coss.shape[0])
    # get an qoss vector that fits the mesh_V scale
    # v_vec = np.linspace(V_min, V_max, int(V_step))
    debug(v_vec)
    qoss = coss_int(v_vec)
    # Scale from pC to nC
    qoss = qoss / 1000

    # np.savetxt('qoss.csv', qoss, delimiter=';')
    return qoss


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
def trial_sim_save():
    """
    Run the complete optimization procedure and save the results in a file
    """
    # Set the basic DAB Specification
    dab_specs = ds.DAB_Specification()
    dab_specs.V1_nom = 700
    dab_specs.V1_min = 600
    dab_specs.V1_max = 800
    # dab_specs.V1_step = math.floor((dab_specs.V1_max - dab_specs.V1_min) / 10 + 1)
    dab_specs.V1_step = 3
    dab_specs.V2_nom = 235
    dab_specs.V2_min = 175
    dab_specs.V2_max = 295
    # dab_specs.V2_step = math.floor((dab_specs.V2_max - dab_specs.V2_min) / 5 + 1)
    dab_specs.V2_step = 4
    dab_specs.P_min = 400
    dab_specs.P_max = 2200
    dab_specs.P_nom = 2000
    # dab_specs.P_step = math.floor((dab_specs.P_max - dab_specs.P_min) / 100 + 1)
    dab_specs.P_step = 5
    dab_specs.n = 2.99
    dab_specs.L_s = 84e-6
    dab_specs.L_m = 599e-6
    dab_specs.fs_nom = 200000

    # Object to store all generated data
    dab_results = ds.DAB_Results()
    # Generate meshes
    dab_results.gen_meshes(
        dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step,
        dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step,
        dab_specs.P_min, dab_specs.P_max, dab_specs.P_step)

    # Modulation Calculation
    # SPS Modulation
    da_mod = mod_sps.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    # TODO where to save??? spec only float...
    simfilepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC_v2.ipes'
    timestep = 100e-12
    simtime = 15e-6

    # Simulation
    da_sim = sim_gecko.start_sim(dab_results.mesh_V1,
                                 dab_results.mesh_V2,
                                 dab_results.mod_sps_phi,
                                 dab_results.mod_sps_tau1,
                                 dab_results.mod_sps_tau2,
                                 simfilepath, timestep, simtime)

    # Simulation
    # Dab_Sim = sim_gecko.Sim_Gecko()
    # da_sim = Dab_Sim.start_sim_threads(dab_results.mesh_V1,
    #                                    dab_results.mesh_V2,
    #                                    dab_results.mod_sps_phi,
    #                                    dab_results.mod_sps_tau1,
    #                                    dab_results.mod_sps_tau2,
    #                                    simfilepath, timestep, simtime)

    # da_sim = Dab_Sim.start_sim_multi(dab_results.mesh_V1,
    #                                  dab_results.mesh_V2,
    #                                  dab_results.mod_sps_phi,
    #                                  dab_results.mod_sps_tau1,
    #                                  dab_results.mod_sps_tau2,
    #                                  simfilepath, timestep, simtime)

    # Unpack the results
    dab_results.append_result_dict(da_sim)

    debug("sim_i_Ls: \n", dab_results.sim_i_Ls)
    debug("sim_S11_p_sw: \n", dab_results.sim_S11_p_sw)

    # Plotting
    info("\nStart Plotting\n")
    plt = plot_dab.Plot_DAB()
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Overview')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.mod_sps_phi[:, 1, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_i_Ls[:, 1, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_S11_p_sw[:, 1, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Power')
    plt.plot_3by1(plt.figs_axes[-1],
                       dab_results.mesh_P[:, 1, :],
                       dab_results.mesh_V2[:, 1, :],
                       dab_results.sim_p_dc1[:, 1, :],
                       dab_results.sim_S11_p_sw[:, 1, :],
                       dab_results.sim_S11_p_cond[:, 1, :],
                       'P / W',
                       'U2 / V',
                       'p_dc1',
                       'S11_p_sw',
                       'S11_p_cond')

    plt.show()

    # Calc power deviation from expected power target
    # power_deviation = mesh_P[vec_vvp].item() and values_mean['mean']['p_dc1'] / mesh_P[vec_vvp].item()
    # debug("power_sim: %f / power_target: %f -> power_deviation: %f" % (values_mean['mean']['p_dc1'], mesh_P[vec_vvp].item(), power_deviation))

    # Saving
    # save_to_file(dab_specs, dab_results, name='mod_sps_sim_v21-v25-p19',
    #              comment='Simulation results for mod_sps with V1 10V res, V2 5V res and P 100W res.')


@timeit
def trail_mod():
    """
    Run the modulation optimization procedure and show the results
    """
    # Set the basic DAB Specification
    dab_specs = ds.DAB_Specification()
    dab_specs.V1_nom = 700
    dab_specs.V1_min = 600
    dab_specs.V1_max = 800
    dab_specs.V1_step = math.floor((dab_specs.V1_max - dab_specs.V1_min) / 10 + 1) # 10V resolution gives 21 steps
    # dab_specs.V1_step = math.floor((dab_specs.V1_max - dab_specs.V1_min) / 10 + 1)
    # dab_specs.V1_step = 1
    dab_specs.V2_nom = 235
    dab_specs.V2_min = 175
    dab_specs.V2_max = 295
    dab_specs.V2_step = math.floor((dab_specs.V2_max - dab_specs.V2_min) / 5 + 1) # 5V resolution gives 25 steps
    # dab_specs.V2_step = math.floor((dab_specs.V2_max - dab_specs.V2_min) / 20 + 1)
    # dab_specs.V2_step = 4
    dab_specs.P_min = 400
    dab_specs.P_max = 2200
    dab_specs.P_nom = 2000
    dab_specs.P_step = math.floor((dab_specs.P_max - dab_specs.P_min) / 100 + 1) # 100W resolution gives 19 steps
    # dab_specs.P_step = math.floor((dab_specs.P_max - dab_specs.P_min) / 300 + 1)
    # dab_specs.P_step = 5
    dab_specs.n = 2.99
    dab_specs.L_s = 84e-6
    dab_specs.L_m = 599e-6
    dab_specs.fs_nom = 200000

    # Set file names
    directory = '~/MA-LEA/LEA/Workdir/dab_optimizer_output/'
    name = 'mod_sps_mcl_v{}-v{}-p{}'.format(int(dab_specs.V1_step),
                                            int(dab_specs.V2_step),
                                            int(dab_specs.P_step))
    if __debug__:
        name = 'debug_' + name
    comment = 'Only modulation calculation results for mod_sps and mod_mcl with V1 {}, V2 {} and P {} steps.'.format(
        int(dab_specs.V1_step),
        int(dab_specs.V2_step),
        int(dab_specs.P_step))
    if __debug__:
        comment = 'Debug ' + comment

    # Object to store all generated data
    dab_results = ds.DAB_Results()
    # Generate meshes
    dab_results.gen_meshes(
        dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step,
        dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step,
        dab_specs.P_min, dab_specs.P_max, dab_specs.P_step)

    # Import Coss curves
    csv_file = '~/MA-LEA/LEA/Files/Datasheets/Coss_C3M0120100J.csv'
    dab_results['coss_C3M0120100J'] = import_Coss(csv_file)
    # Generate Qoss matrix
    dab_results['qoss_C3M0120100J'] = integrate_Coss(dab_results['coss_C3M0120100J'])

    # Modulation Calculation
    # ZVS Modulation
    da_mod = mod_zvs.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    # debug(dab_results)

    # Plotting
    info("\nStart Plotting\n")
    v1_middle = int(np.shape(dab_results.mesh_P)[1] / 2)
    debug('View plane: U_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab_results.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB()

    # Plot Coss
    plt.new_fig(nrows=1, ncols=2, tab_title='Coss C3M0120100J', sharex=False, sharey=False)
    plt.subplot(np.arange(dab_results['coss_C3M0120100J'].shape[0]),
                     dab_results['coss_C3M0120100J'],
                     ax=plt.figs_axes[-1][1][0],
                     xlabel='U_DS / V', ylabel='C_oss / pF', title='Coss C3M0120100J',
                     yscale='log')
    plt.subplot(np.arange(dab_results['qoss_C3M0120100J'].shape[0]),
                     dab_results['qoss_C3M0120100J'],
                     ax=plt.figs_axes[-1][1][1],
                     xlabel='U_DS / V', ylabel='Q_oss / nC', title='Qoss C3M0120100J')

    # Plot OptZVS mod results
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='OptZVS Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_zvs_phi[:, v1_middle, :],
                             dab_results.mod_zvs_tau1[:, v1_middle, :],
                             dab_results.mod_zvs_tau2[:, v1_middle, :],
                             mask1=dab_results.mod_zvs_mask_m1p[:, v1_middle, :],
                             mask2=dab_results.mod_zvs_mask_m2[:, v1_middle, :],
                             maskZVS=dab_results.mod_zvs_mask_zvs[:, v1_middle, :]
                             )

    plt.show()


@timeit
def dab_mod_save():
    """
    Run the modulation optimization procedure and save the results in a file
    """
    # Set the basic DAB Specification
    dab_specs = ds.DAB_Specification()
    dab_specs.V1_nom = 700
    dab_specs.V1_min = 700
    dab_specs.V1_max = 700
    # dab_specs.V1_step = math.floor((dab_specs.V1_max - dab_specs.V1_min) / 10 + 1) # 10V resolution gives 21 steps
    # dab_specs.V1_step = math.floor((dab_specs.V1_max - dab_specs.V1_min) / 10 + 1)
    dab_specs.V1_step = 1
    dab_specs.V2_nom = 235
    dab_specs.V2_min = 175
    dab_specs.V2_max = 295
    # dab_specs.V2_step = math.floor((dab_specs.V2_max - dab_specs.V2_min) / 5 + 1) # 5V resolution gives 25 steps
    dab_specs.V2_step = math.floor((dab_specs.V2_max - dab_specs.V2_min) / 20 + 1)
    # dab_specs.V2_step = 4
    dab_specs.P_min = 400
    dab_specs.P_max = 2200
    dab_specs.P_nom = 2000
    # dab_specs.P_step = math.floor((dab_specs.P_max - dab_specs.P_min) / 100 + 1) # 100W resolution gives 19 steps
    dab_specs.P_step = math.floor((dab_specs.P_max - dab_specs.P_min) / 300 + 1)
    # dab_specs.P_step = 5
    dab_specs.n = 2.99
    dab_specs.L_s = 84e-6
    dab_specs.L_m = 599e-6
    dab_specs.fs_nom = 200000

    # Set file names
    directory = '~/MA-LEA/LEA/Workdir/dab_optimizer_output/'
    name = 'mod_sps_mcl_v{}-v{}-p{}'.format(int(dab_specs.V1_step),
                                            int(dab_specs.V2_step),
                                            int(dab_specs.P_step))
    if __debug__:
        name = 'debug_' + name
    comment = 'Only modulation calculation results for mod_sps and mod_mcl with V1 {}, V2 {} and P {} steps.'.format(
        int(dab_specs.V1_step),
        int(dab_specs.V2_step),
        int(dab_specs.P_step))
    if __debug__:
        comment = 'Debug ' + comment

    # Object to store all generated data
    dab_results = ds.DAB_Results()
    # Generate meshes
    dab_results.gen_meshes(
        dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step,
        dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step,
        dab_specs.P_min, dab_specs.P_max, dab_specs.P_step)

    # Import Coss curves
    csv_file = '~/MA-LEA/LEA/Files/Datasheets/Coss_C3M0120100J.csv'
    import_Coss(dab_results, csv_file, 'C3M0120100J')
    debug(dab_results)

    # Modulation Calculation
    # SPS Modulation
    da_mod = mod_sps.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    # Modulation Calculation
    # MCL Modulation
    da_mod = mod_mcl.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    # Saving
    # Create new dir for all files
    directory = directory + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    os.mkdir(directory)
    # Save data
    save_to_file(dab_specs, dab_results, directory=directory, name=name, timestamp=False, comment=comment)

    # Open existing file and export array to csv
    file = name
    # Loading
    dab_file = '~/MA-LEA/LEA/Workdir/dab_optimizer_output/{0}/{0}.npz'.format(file)
    dab_file = os.path.join(directory, file) + '.npz'
    dab_file = os.path.expanduser(dab_file)
    dab_file = os.path.expandvars(dab_file)
    dab_file = os.path.abspath(dab_file)
    dab_specs, dab_results = load_from_file(dab_file)
    # results key:
    keys = ['mod_sps_phi', 'mod_sps_tau1', 'mod_sps_tau2', 'mod_mcl_phi', 'mod_mcl_tau1', 'mod_mcl_tau2']
    # Convert phi, tau1/2 from rad to duty cycle * 10000
    # This is for old saved results where phi,tau is in rad
    # But in DAB-Controller we need duty cycle * 10000 (2pi eq. 10000)
    for key in keys:
        dab_results[key] = dab_results[key] / (2 * np.pi) * 10000
    # Set file names
    directory = os.path.dirname(dab_file)
    file = os.path.basename(dab_file)
    name = os.path.splitext(file.split('_', 2)[2])[0]
    for key in keys:
        save_to_csv(dab_specs, dab_results, key, directory, name)

    # Plotting
    info("\nStart Plotting\n")
    v1_middle = int(np.shape(dab_results.mesh_P)[1] / 2)
    debug('View plane: U_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab_results.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB()

    # Plot SPS sim results
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_sps_phi[:, v1_middle, :],
                             dab_results.mod_sps_tau1[:, v1_middle, :],
                             dab_results.mod_sps_tau2[:, v1_middle, :],
                             # mask1=dab_results.mod_sps_mask_tcm[:, v1_middle, :],
                             # mask2=dab_results.mod_sps_mask_cpm[:, v1_middle, :]
                             )

    # Plot MCL sim results
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_mcl_phi[:, v1_middle, :],
                             dab_results.mod_mcl_tau1[:, v1_middle, :],
                             dab_results.mod_mcl_tau2[:, v1_middle, :],
                             mask1=dab_results.mod_mcl_mask_tcm[:, v1_middle, :],
                             mask2=dab_results.mod_mcl_mask_cpm[:, v1_middle, :]
                             )

    # Save plots
    metadata = {'Title':       name,
                'Description': comment,
                'Author':      'Felix Langemeier',
                'Software':    'python, matplotlib'}
    # The PNG specification defines some common keywords:
    # Title	Short (one line) title or caption for image
    # Author	Name of image's creator
    # Description	Description of image (possibly long)
    # Copyright	Copyright notice
    # Creation Time	Time of original image creation
    # Software	Software used to create the image
    # Disclaimer	Legal disclaimer
    # Warning	Warning of nature of content
    # Source	Device used to create the image
    # Comment	Miscellaneous comment
    i = 0
    for fig in plt.figs_axes:
        fname = os.path.join(directory + '/' + name + '_fig{:0>2d}.png'.format(i))
        fig[0].savefig(fname=fname, metadata=metadata)
        i += 1
    # TODO Fix that the first and following image sizes differ. First is window size, following are 1000x500px.

    plt.show()


@timeit
def dab_sim_save():
    """
    Run the complete optimization procedure and save the results in a file
    """
    # Set the basic DAB Specification
    dab_specs = ds.DAB_Specification()
    dab_specs.V1_nom = 700
    dab_specs.V1_min = 600
    dab_specs.V1_max = 800
    # dab_specs.V1_step = math.floor((dab_specs.V1_max - dab_specs.V1_min) / 10 + 1) # 10V resolution gives 21 steps
    # dab_specs.V1_step = math.floor((dab_specs.V1_max - dab_specs.V1_min) / 10 + 1)
    dab_specs.V1_step = 3
    dab_specs.V2_nom = 235
    dab_specs.V2_min = 175
    dab_specs.V2_max = 295
    # dab_specs.V2_step = math.floor((dab_specs.V2_max - dab_specs.V2_min) / 5 + 1) # 5V resolution gives 25 steps
    dab_specs.V2_step = math.floor((dab_specs.V2_max - dab_specs.V2_min) / 5 + 1)
    # dab_specs.V2_step = 4
    dab_specs.P_min = 400
    dab_specs.P_max = 2200
    dab_specs.P_nom = 2000
    # dab_specs.P_step = math.floor((dab_specs.P_max - dab_specs.P_min) / 100 + 1) # 100W resolution gives 19 steps
    dab_specs.P_step = math.floor((dab_specs.P_max - dab_specs.P_min) / 100 + 1)
    # dab_specs.P_step = 5
    dab_specs.n = 2.99
    dab_specs.L_s = 84e-6
    dab_specs.L_m = 599e-6
    dab_specs.fs_nom = 200000

    # TODO where to save??? spec only float...
    # Set sim defaults
    simfilepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC_v2.ipes'
    timestep = 100e-12
    simtime = 15e-6
    geckoport = 43036
    # Set file names
    directory = '~/MA-LEA/LEA/Workdir/dab_optimizer_output/'
    name = 'mod_sps_mcl_sim_Gv2_L84_v{}-v{}-p{}'.format(int(dab_specs.V1_step),
                                                        int(dab_specs.V2_step),
                                                        int(dab_specs.P_step))
    if __debug__:
        name = 'debug_' + name
    comment = 'Simulation results for mod_sps and mod_mcl with V1 {}, V2 {} and P {} steps.'.format(
        int(dab_specs.V1_step),
        int(dab_specs.V2_step),
        int(dab_specs.P_step))
    comment = comment + '\n' + 'Using simfilepath = ' + simfilepath
    if __debug__:
        comment = 'Debug ' + comment

    # Object to store all generated data
    dab_results = ds.DAB_Results()
    # Generate meshes
    dab_results.gen_meshes(
        dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step,
        dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step,
        dab_specs.P_min, dab_specs.P_max, dab_specs.P_step)

    # Modulation Calculation
    # SPS Modulation
    da_mod = mod_sps.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    # Simulation
    da_sim = sim_gecko.start_sim(dab_results.mesh_V1,
                                 dab_results.mesh_V2,
                                 dab_results.mod_sps_phi,
                                 dab_results.mod_sps_tau1,
                                 dab_results.mod_sps_tau2,
                                 simfilepath, timestep, simtime, geckoport=geckoport)

    # Unpack the results
    dab_results.append_result_dict(da_sim, name_pre='sim_sps_')

    # Calc power deviation from expected power target
    sim_sps_power_deviation = dab_results.sim_sps_p_dc1 / dab_results.mesh_P - 1

    # Modulation Calculation
    # MCL Modulation
    da_mod = mod_mcl.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    # Simulation
    da_sim = sim_gecko.start_sim(dab_results.mesh_V1,
                                 dab_results.mesh_V2,
                                 dab_results.mod_mcl_phi,
                                 dab_results.mod_mcl_tau1,
                                 dab_results.mod_mcl_tau2,
                                 simfilepath, timestep, simtime, geckoport=geckoport)

    # Unpack the results
    dab_results.append_result_dict(da_sim, name_pre='sim_mcl_')

    # Calc power deviation from expected power target
    sim_mcl_power_deviation = dab_results.sim_mcl_p_dc1 / dab_results.mesh_P - 1

    # Saving
    # Create new dir for all files
    directory = directory + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    os.mkdir(directory)
    # Save data
    save_to_file(dab_specs, dab_results, directory=directory, name=name, comment=comment)

    # Plotting
    info("\nStart Plotting\n")
    v1_middle = int(np.shape(dab_results.mesh_P)[1] / 2)
    debug('View plane: U_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab_results.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB()

    # Plot SPS sim results
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Overview')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.mod_sps_phi[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_i_Ls[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_S11_p_sw[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    # Plot power loss
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Power')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                           dab_results.mesh_V2[:, v1_middle, :],
    #                           dab_results.sim_sps_p_dc1[:, v1_middle, :],
    #                           ax=plt.figs_axes[-1][1][0],
    #                           xlabel='P / W', ylabel='U2 / V', title='sim_sps_p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              sim_sps_power_deviation[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='power deviation')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_S11_p_sw[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_S11_p_cond[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_cond / W')

    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_sps_phi[:, v1_middle, :],
                             dab_results.mod_sps_tau1[:, v1_middle, :],
                             dab_results.mod_sps_tau2[:, v1_middle, :],
                             # mask1=dab_results.mod_sps_mask_tcm[:, v1_middle, :],
                             # mask2=dab_results.mod_sps_mask_cpm[:, v1_middle, :]
                             )

    # Plot MCL sim results
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Overview')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.mod_mcl_phi[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_i_Ls[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_S11_p_sw[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    # Plot power loss
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Power')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                           dab_results.mesh_V2[:, v1_middle, :],
    #                           dab_results.sim_mcl_p_dc1[:, v1_middle, :],
    #                           ax=plt.figs_axes[-1][1][0],
    #                           xlabel='P / W', ylabel='U2 / V', title='sim_mcl_p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              sim_mcl_power_deviation[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='power deviation')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_S11_p_sw[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_S11_p_cond[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_cond / W')

    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_mcl_phi[:, v1_middle, :],
                             dab_results.mod_mcl_tau1[:, v1_middle, :],
                             dab_results.mod_mcl_tau2[:, v1_middle, :],
                             mask1=dab_results.mod_mcl_mask_tcm[:, v1_middle, :],
                             mask2=dab_results.mod_mcl_mask_cpm[:, v1_middle, :]
                             )

    # Save plots
    metadata = {'Title':       name,
                'Description': comment,
                'Author':      'Felix Langemeier',
                'Software':    'python, matplotlib'}
    # The PNG specification defines some common keywords:
    # Title	Short (one line) title or caption for image
    # Author	Name of image's creator
    # Description	Description of image (possibly long)
    # Copyright	Copyright notice
    # Creation Time	Time of original image creation
    # Software	Software used to create the image
    # Disclaimer	Legal disclaimer
    # Warning	Warning of nature of content
    # Source	Device used to create the image
    # Comment	Miscellaneous comment
    i = 0
    for fig in plt.figs_axes:
        fname = os.path.join(directory + '/' + name + '_fig{:0>2d}.png'.format(i))
        fig[0].savefig(fname=fname, metadata=metadata)
        i += 1
    # TODO Fix that the first and following image sizes differ. First is window size, following are 1000x500px.

    plt.show()


@timeit
def trial_dab():
    """
    Run the complete optimization procedure
    """
    # Set the basic DAB Specification
    # Setting it this way enables tab completion.
    dab_specs = ds.DAB_Specification()
    dab_specs.V1_nom = 700
    dab_specs.V1_min = 600
    dab_specs.V1_max = 800
    dab_specs.V1_step = 2
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
    # SPS Modulation
    da_mod = mod_sps.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    # TODO where to save??? spec only float...
    simfilepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC_v2.ipes'
    timestep = 100e-12
    simtime = 15e-6

    # Simulation
    d_sim = sim_gecko.start_sim(dab_results.mesh_V1,
                                dab_results.mesh_V2,
                                dab_results.mod_sps_phi,
                                dab_results.mod_sps_tau1,
                                dab_results.mod_sps_tau2,
                                simfilepath, timestep, simtime)

    # Unpack the results
    # TODO maybe put this as a function in DAB_Results
    for k, v in d_sim.items():
        dab_results['sim_' + k] = v

    debug("sim_i_Ls: \n", dab_results.sim_i_Ls)
    debug("sim_S11_p_sw: \n", dab_results.sim_S11_p_sw)

    # Plotting
    info("\nStart Plotting\n")
    plt = plot_dab.Plot_DAB()
    # Mod SPS
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Overview')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.mod_sps_phi[:, 1, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_i_Ls[:, 1, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_S11_p_sw[:, 1, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Power')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_p_dc1[:, 1, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_S11_p_sw[:, 1, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_S11_p_cond[:, 1, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_cond / W')

    # Mod MCL
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Modulation')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, 1, :],
                             dab_results.mesh_V2[:, 1, :],
                             dab_results.sim_p_dc1[:, 1, :],
                             dab_results.sim_S11_p_sw[:, 1, :],
                             dab_results.sim_S11_p_cond[:, 1, :])

    plt.show()

    # Saving
    save_to_file(dab_specs, dab_results, directory='~/MA-LEA/LEA/Workdir/dab_optimizer_output', name='test-save',
                 comment='This is a saving test with random data!')
    # save_to_file(dab_specs, dab_results, name='test-save', timestamp=False, comment='This is a saving test with random data!')

    # Loading
    # dab_specs_loaded, dab_results_loaded = load_from_file('test-save.npz')
    # dab_specs_loaded.pprint()
    # dab_results_loaded.pprint()

    # add some false data, should output an error log or warning
    # dab_results_loaded.foo = np.array([1, 2, 3])
    # dab_results_loaded.bar = "test"


@timeit
def trial_plot_modresults():
    """
    Run the modulation optimization procedure and plot the results
    """
    # Set the basic DAB Specification
    dab_specs = ds.DAB_Specification()
    dab_specs.V1_nom = 700
    dab_specs.V1_min = 600
    dab_specs.V1_max = 800
    dab_specs.V1_step = 21 * 3
    dab_specs.V2_nom = 235
    dab_specs.V2_min = 175
    dab_specs.V2_max = 295
    dab_specs.V2_step = 25 * 3
    dab_specs.P_min = 400
    dab_specs.P_max = 2200
    dab_specs.P_nom = 2000
    dab_specs.P_step = 19 * 3
    dab_specs.n = 2.99
    dab_specs.L_s = 84e-6
    dab_specs.L_m = 599e-6
    dab_specs.fs_nom = 200000

    # Object to store all generated data
    dab_results = ds.DAB_Results()
    # gen mesh
    dab_results.gen_meshes(
        dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step,
        dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step,
        dab_specs.P_min, dab_specs.P_max, dab_specs.P_step)

    # Modulation Calculation
    # SPS Modulation
    da_mod = mod_sps.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    # Modulation Calculation
    # SPS Modulation
    da_mod = mod_mcl.calc_modulation(dab_specs.n,
                                     dab_specs.L_s,
                                     dab_specs.fs_nom,
                                     dab_results.mesh_V1,
                                     dab_results.mesh_V2,
                                     dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    info("\nStart Plotting\n")

    v1_middle = int(np.shape(dab_results.mesh_P)[1] / 2)

    plt = plot_dab.Plot_DAB()
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Modulation Angles')
    # plt.plot_modulation(plt.figs_axes[-1],
    #                          dab_results.mesh_P[:, v1_middle, :],
    #                          dab_results.mesh_V2[:, v1_middle, :],
    #                          dab_results.mod_mcl_phi[:, v1_middle, :],
    #                          dab_results.mod_mcl_tau1[:, v1_middle, :],
    #                          dab_results.mod_mcl_tau2[:, v1_middle, :],
    #                          mask1=dab_results.mod_mcl_mask_tcm[:, v1_middle, :],
    #                          mask2=dab_results.mod_mcl_mask_cpm[:, v1_middle, :])

    # Plot animation for every V1 cross-section
    # for v1 in range(0, np.shape(dab_results.mesh_P)[1] - 1):
    #     print(v1)
    #     plt.plot_modulation(plt.figs_axes[-1],
    #                              dab_results.mesh_P[:, v1, :],
    #                              dab_results.mesh_V2[:, v1, :],
    #                              dab_results.mod_mcl_phi[:, v1, :],
    #                              dab_results.mod_mcl_tau1[:, v1, :],
    #                              dab_results.mod_mcl_tau2[:, v1, :],
    #                              mask1=dab_results.mod_mcl_mask_tcm[:, v1, :],
    #                              mask2=dab_results.mod_mcl_mask_cpm[:, v1, :])

    import matplotlib.animation as animation

    def animate(v1, fig_axes: tuple, x, y, z1, z2, z3, mask1=None, mask2=None, mask3=None):
        print(v1, dab_results.mesh_V1[0, v1, 0])
        plt.latex = True
        if plt.latex:
            title = '$U_1 = {:.1f}'.format(dab_results.mesh_V1[0, v1, 0]) + '\mathrm{V}$'
        else:
            title = 'U_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1, 0])
        plt.plot_modulation(fig_axes,
                                 x[:, v1, :],
                                 y[:, v1, :],
                                 z1[:, v1, :],
                                 z2[:, v1, :],
                                 z3[:, v1, :],
                                 title,
                                 mask1[:, v1, :],
                                 mask2[:, v1, :])
        return fig_axes[1]

    args = (plt.figs_axes[-1],
            dab_results.mesh_P,
            dab_results.mesh_V2,
            dab_results.mod_mcl_phi,
            dab_results.mod_mcl_tau1,
            dab_results.mod_mcl_tau2,
            dab_results.mod_mcl_mask_tcm,
            dab_results.mod_mcl_mask_cpm)

    ani = animation.FuncAnimation(
        plt.figs_axes[-1][0],
        animate,
        frames=np.shape(dab_results.mesh_P)[1] - 1,
        fargs=args,
        blit=False,  # blitting can't be used with Figure artists
    )

    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save('animation.mp4', writer=FFwriter)

    # Plot SPS modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_sps_phi[:, v1_middle, :],
                             dab_results.mod_sps_tau1[:, v1_middle, :],
                             dab_results.mod_sps_tau2[:, v1_middle, :])

    plt.show()


def trial_plot_simresults():
    # Loading
    dab_specs, dab_results = load_from_file('~/MA-LEA/LEA/Workdir/dab_optimizer_output/test-sps-save.npz')
    dab_specs.pprint()
    # dab_results.pprint()

    info("\nStart Plotting\n")
    plt = plot_dab.Plot_DAB()
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Overview')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.mod_sps_phi[:, 1, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_i_Ls[:, 1, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_S11_p_sw[:, 1, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Power')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_p_dc1[:, 1, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='sim_p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_S11_p_sw[:, 1, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')
    plt.subplot_contourf(dab_results.mesh_P[:, 1, :],
                              dab_results.mesh_V2[:, 1, :],
                              dab_results.sim_S11_p_cond[:, 1, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_cond / W')
    # plt.plot_3by1(plt.figs_axes[-1],
    #                   dab_results.mesh_P[:, 1, :],
    #                   dab_results.mesh_V2[:, 1, :],
    #                   dab_results.sim_p_dc1[:, 1, :],
    #                   dab_results.sim_S11_p_sw[:, 1, :],
    #                   dab_results.sim_S11_p_cond[:, 1, :],
    #                   'P / W',
    #                   'U2 / V',
    #                   'p_dc1',
    #                   'S11_p_sw',
    #                   'S11_p_cond')

    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Modulation')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, 1, :],
                             dab_results.mesh_V2[:, 1, :],
                             dab_results.mod_sps_phi[:, 1, :],
                             dab_results.mod_sps_tau1[:, 1, :],
                             dab_results.mod_sps_tau2[:, 1, :])

    # now redraw the previous fig
    # plt.plot_modulation(plt.figs_axes[-1],
    #                         dab_results.mesh_P[:, 1, :],
    #                         dab_results.mesh_V2[:, 1, :],
    #                         dab_results.mod_sps_phi[:, 1, :],
    #                         dab_results.mod_sps_phi[:, 1, :],
    #                         dab_results.mod_sps_tau2[:, 1, :])

    plt.show()


def plot_simresults():
    # Select File and V1 Plane
    file = '2023-01-26_15:50:29_mod_sps_mcl_sim_L84_v3-v25-p19'
    # V1 index
    v1 = None
    # v1 = 0

    # Loading
    dab_file = '~/MA-LEA/LEA/Workdir/dab_optimizer_output/{0}/{0}.npz'.format(file)
    dab_file = os.path.expanduser(dab_file)
    dab_file = os.path.expandvars(dab_file)
    dab_file = os.path.abspath(dab_file)
    dab_specs, dab_results = load_from_file(dab_file)
    dab_specs.pprint()
    # dab_results.pprint()

    # Set file names
    directory = os.path.dirname(dab_file)
    file = os.path.basename(dab_file)
    name = os.path.splitext(file.split('_', 2)[2])[0]

    comment = str(dab_results._comment)

    # Calc power deviation from expected power target
    sim_sps_power_deviation = dab_results.sim_sps_p_dc1 / dab_results.mesh_P - 1
    sim_mcl_power_deviation = dab_results.sim_mcl_p_dc1 / dab_results.mesh_P - 1

    # Plotting
    info("\nStart Plotting\n")
    v1_middle = v1 if v1 is not None else int(np.shape(dab_results.mesh_P)[1] / 2)
    debug('View plane: U_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab_results.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab_results.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB()

    # Plot SPS sim results
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Overview')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.mod_sps_phi[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_i_Ls[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_S11_p_sw[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    # Plot power loss
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Power')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                           dab_results.mesh_V2[:, v1_middle, :],
    #                           dab_results.sim_sps_p_dc1[:, v1_middle, :],
    #                           ax=plt.figs_axes[-1][1][0],
    #                           xlabel='P / W', ylabel='U2 / V', title='sim_sps_p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              sim_sps_power_deviation[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='power deviation')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_S11_p_sw[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_S11_p_cond[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_cond / W')

    # Plot power loss
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Power2')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                           dab_results.mesh_V2[:, v1_middle, :],
    #                           dab_results.sim_sps_p_dc1[:, v1_middle, :],
    #                           ax=plt.figs_axes[-1][1][0],
    #                           xlabel='P / W', ylabel='U2 / V', title='sim_sps_p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              sim_sps_power_deviation[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='power deviation')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_sps_p_dc1[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.mod_sps_phi[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='phi / rad')

    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_sps_phi[:, v1_middle, :],
                             dab_results.mod_sps_tau1[:, v1_middle, :],
                             dab_results.mod_sps_tau2[:, v1_middle, :],
                             # mask1=dab_results.mod_sps_mask_tcm[:, v1_middle, :],
                             # mask2=dab_results.mod_sps_mask_cpm[:, v1_middle, :]
                             )

    # Plot MCL sim results
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Overview')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.mod_mcl_phi[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_i_Ls[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='i_Ls / A')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_S11_p_sw[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')

    # Plot power loss
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Power')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                           dab_results.mesh_V2[:, v1_middle, :],
    #                           dab_results.sim_mcl_p_dc1[:, v1_middle, :],
    #                           ax=plt.figs_axes[-1][1][0],
    #                           xlabel='P / W', ylabel='U2 / V', title='sim_mcl_p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              sim_mcl_power_deviation[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='power deviation')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_S11_p_sw[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_sw / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_S11_p_cond[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='S11_p_cond / W')

    # Plot power loss
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Power2')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                           dab_results.mesh_V2[:, v1_middle, :],
    #                           dab_results.sim_mcl_p_dc1[:, v1_middle, :],
    #                           ax=plt.figs_axes[-1][1][0],
    #                           xlabel='P / W', ylabel='U2 / V', title='sim_mcl_p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              sim_mcl_power_deviation[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][0],
                              xlabel='P / W', ylabel='U2 / V', title='power deviation')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.sim_mcl_p_dc1[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][1],
                              xlabel='P / W', ylabel='U2 / V', title='p_dc1 / W')
    plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
                              dab_results.mesh_V2[:, v1_middle, :],
                              dab_results.mod_mcl_phi[:, v1_middle, :],
                              ax=plt.figs_axes[-1][1][2],
                              xlabel='P / W', ylabel='U2 / V', title='phi / rad')

    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_mcl_phi[:, v1_middle, :],
                             dab_results.mod_mcl_tau1[:, v1_middle, :],
                             dab_results.mod_mcl_tau2[:, v1_middle, :],
                             mask1=dab_results.mod_mcl_mask_tcm[:, v1_middle, :],
                             mask2=dab_results.mod_mcl_mask_cpm[:, v1_middle, :]
                             )

    # Save plots
    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
    metadata = {'Title':       name,
                'Description': comment,
                'Author':      'Felix Langemeier',
                'Software':    'python, matplotlib'}
    # The PNG specification defines some common keywords:
    # Title	Short (one line) title or caption for image
    # Author	Name of image's creator
    # Description	Description of image (possibly long)
    # Copyright	Copyright notice
    # Creation Time	Time of original image creation
    # Software	Software used to create the image
    # Disclaimer	Legal disclaimer
    # Warning	Warning of nature of content
    # Source	Device used to create the image
    # Comment	Miscellaneous comment
    i = 0
    for fig in plt.figs_axes:
        fname = os.path.join(directory + '/' + name + '_fig{:0>2d}.png'.format(i))
        if __debug__:
            debug(fname, metadata)
        else:
            fig[0].savefig(fname=fname, metadata=metadata)
        i += 1
    # TODO Fix that the first and following image sizes differ. First is window size, following are 1000x500px.

    plt.show()


def _main_dummy():
    return


# ---------- MAIN ----------
if __name__ == '__main__':
    info("Start of DAB Optimizer ...")
    # Do some basic init like logging, args, etc.
    main_init()

    # Only modulation calculation
    trail_mod()
    # dab_mod_save()

    # Generate simulation data
    # dab_sim_save()
    # trial_sim_save()

    # Test the DAB functions
    # test_dab()
    # Test the Plot functions
    # trial_plot_simresults()
    # trial_plot_modresults()

    # Plot saved results
    # plot_simresults()

    # # Open existing file and export array to csv
    # file = '2023-04-03_04:57:55_mod_sps_mcl_sim_Gv2_L84_v3-v25-p19'
    # # Loading
    # dab_file = '~/MA-LEA/LEA/Workdir/dab_optimizer_output/{0}/{0}.npz'.format(file)
    # dab_file = os.path.expanduser(dab_file)
    # dab_file = os.path.expandvars(dab_file)
    # dab_file = os.path.abspath(dab_file)
    # dab_specs, dab_results = load_from_file(dab_file)
    # # results key:
    # keys = ['mod_sps_phi', 'mod_sps_tau1', 'mod_sps_tau2', 'mod_mcl_phi', 'mod_mcl_tau1', 'mod_mcl_tau2']
    # # Convert phi, tau1/2 from rad to duty cycle * 10000
    # # This is for old saved results where phi,tau is in rad
    # # But in DAB-Controller we need duty cycle * 10000 (2pi eq. 10000)
    # for key in keys:
    #     dab_results[key] = dab_results[key] / (2 * np.pi) * 10000
    # # Set file names
    # directory = os.path.dirname(dab_file)
    # file = os.path.basename(dab_file)
    # name = os.path.splitext(file.split('_', 2)[2])[0]
    # for key in keys:
    #     save_to_csv(dab_specs, dab_results, key, directory, name)

    # sys.exit(0)
