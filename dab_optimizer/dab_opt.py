#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10
import copy
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


def integrate_Coss(coss):
    """
    Integrate Coss for each voltage from 0 to V_max
    :param coss: MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
    :return: Qoss(Vds) as one row of data and index = Vds.
    """

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
def dab_mod_save():
    """
    Run the modulation optimization procedure and save the results in a file
    """
    # Set the basic DAB Specification
    dab = ds.DAB_Data()
    dab.V1_nom = 700
    dab.V1_min = 700
    dab.V1_max = 700
    # dab.V1_step = math.floor((dab.V1_max - dab.V1_min) / 10 + 1) # 10V resolution gives 21 steps
    # dab.V1_step = math.floor((dab.V1_max - dab.V1_min) / 10 + 1)
    dab.V1_step = 1
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    # dab.V2_step = math.floor((dab.V2_max - dab.V2_min) / 5 + 1) # 5V resolution gives 25 steps
    dab.V2_step = math.floor((dab.V2_max - dab.V2_min) / 20 + 1)
    # dab.V2_step = 4
    dab.P_min = 400
    dab.P_max = 2200
    dab.P_nom = 2000
    # dab.P_step = math.floor((dab.P_max - dab.P_min) / 100 + 1) # 100W resolution gives 19 steps
    dab.P_step = math.floor((dab.P_max - dab.P_min) / 300 + 1)
    # dab.P_step = 5
    dab.n = 2.99
    # Values for mod ZVS
    dab.Ls = 83e-6
    dab.Lm = 595e-6
    # dab.Lc1 = 25.62e-3
    # Assumption for tests
    dab.Lc1 = 611e-6
    dab.Lc2 = 611e-6
    dab.fs = 200000
    # Values for mod sps, mcl
    dab.L_s = 83e-6
    dab.L_m = 595e-6
    dab.fs_nom = 200000
    # Generate meshes
    dab.gen_meshes()

    # Import Coss curves
    csv_file = '~/MA-LEA/LEA/Files/Datasheets/Coss_C3M0120100J.csv'
    dab['coss_C3M0120100J'] = import_Coss(csv_file)
    # Generate Qoss matrix
    dab['qoss_C3M0120100J'] = integrate_Coss(dab['coss_C3M0120100J'])

    # Set file names
    directory = '~/MA-LEA/LEA/Workdir/dab_optimizer_output/'
    name = 'mod_sps_mcl_v{}-v{}-p{}'.format(int(dab.V1_step),
                                            int(dab.V2_step),
                                            int(dab.P_step))
    if __debug__:
        name = 'debug_' + name
    comment = 'Only modulation calculation results for mod_sps and mod_mcl with V1 {}, V2 {} and P {} steps.'.format(
        int(dab.V1_step),
        int(dab.V2_step),
        int(dab.P_step))
    if __debug__:
        comment = 'Debug ' + comment

    # Full directory name with timestamp
    directory = directory + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name

    # Modulation Calculation
    # SPS Modulation
    da_mod = mod_sps.calc_modulation(dab.n,
                                     dab.L_s,
                                     dab.fs_nom,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)

    # Unpack the results
    dab.append_result_dict(da_mod)

    # Modulation Calculation
    # MCL Modulation
    da_mod = mod_mcl.calc_modulation(dab.n,
                                     dab.L_s,
                                     dab.fs_nom,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)

    # Unpack the results
    dab.append_result_dict(da_mod)

    # Saving
    # Create new dir for all files
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    os.mkdir(directory)
    # Save data
    ds.save_to_file(dab, directory=directory, name=name, timestamp=False, comment=comment)

    # Save to csv for DAB-Controller
    # Meshes to save:
    keys = ['mod_sps_phi', 'mod_sps_tau1', 'mod_sps_tau2', 'mod_mcl_phi', 'mod_mcl_tau1', 'mod_mcl_tau2']
    # Convert phi, tau1/2 from rad to duty cycle * 10000
    # In DAB-Controller we need duty cycle * 10000 (2pi eq. 10000)
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 10000
        ds.save_to_csv(dab2, key, directory, name, timestamp=False)

    # Plotting
    info("\nStart Plotting\n")
    v1_middle = int(np.shape(dab.mesh_P)[1] / 2)
    debug('View plane: U_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB()

    # Plot SPS sim results
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='SPS Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                        dab.mesh_P[:, v1_middle, :],
                        dab.mesh_V2[:, v1_middle, :],
                        dab.mod_sps_phi[:, v1_middle, :],
                        dab.mod_sps_tau1[:, v1_middle, :],
                        dab.mod_sps_tau2[:, v1_middle, :],
                        # mask1=dab.mod_sps_mask_tcm[:, v1_middle, :],
                        # mask2=dab.mod_sps_mask_cpm[:, v1_middle, :]
                        )

    # Plot MCL sim results
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='MCL Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                        dab.mesh_P[:, v1_middle, :],
                        dab.mesh_V2[:, v1_middle, :],
                        dab.mod_mcl_phi[:, v1_middle, :],
                        dab.mod_mcl_tau1[:, v1_middle, :],
                        dab.mod_mcl_tau2[:, v1_middle, :],
                        mask1=dab.mod_mcl_mask_tcm[:, v1_middle, :],
                        mask2=dab.mod_mcl_mask_cpm[:, v1_middle, :]
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
    ds.old_save_to_file(dab_specs, dab_results, directory=directory, name=name, comment=comment)

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
def trail_mod():
    """
    Run the modulation optimization procedure and show the results
    """
    # Set the basic DAB Specification
    dab = ds.DAB_Data()
    dab.V1_nom = 700
    dab.V1_min = 600
    dab.V1_max = 800
    dab.V1_step = math.floor((dab.V1_max - dab.V1_min) / 10 + 1)  # 10V resolution gives 21 steps
    # dab.V1_step = math.floor((dab.V1_max - dab.V1_min) / 10 + 1)
    # dab.V1_step = 1
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = math.floor((dab.V2_max - dab.V2_min) / 5 + 1)  # 5V resolution gives 25 steps
    # dab.V2_step = math.floor((dab.V2_max - dab.V2_min) / 20 + 1)
    # dab.V2_step = 4
    dab.P_min = 400
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = math.floor((dab.P_max - dab.P_min) / 100 + 1)  # 100W resolution gives 19 steps
    # dab.P_step = math.floor((dab.P_max - dab.P_min) / 300 + 1)
    # dab.P_step = 5
    dab.n = 2.99
    dab.Ls = 83e-6
    dab.Lm = 595e-6
    # dab.Lc1 = 25.62e-3
    # Assumption for tests
    dab.Lc1 = 611e-6
    dab.Lc2 = 611e-6
    dab.fs = 200000
    # Generate meshes
    dab.gen_meshes()

    # Set file names
    directory = '~/MA-LEA/LEA/Workdir/dab_optimizer_output/'
    name = 'mod_sps_mcl_v{}-v{}-p{}'.format(int(dab.V1_step),
                                            int(dab.V2_step),
                                            int(dab.P_step))
    if __debug__:
        name = 'debug_' + name
    comment = 'Only modulation calculation results for mod_sps and mod_mcl with V1 {}, V2 {} and P {} steps.'.format(
        int(dab.V1_step),
        int(dab.V2_step),
        int(dab.P_step))
    if __debug__:
        comment = 'Debug ' + comment

    # Import Coss curves
    csv_file = '~/MA-LEA/LEA/Files/Datasheets/Coss_C3M0120100J.csv'
    dab['coss_C3M0120100J'] = import_Coss(csv_file)
    # Generate Qoss matrix
    dab['qoss_C3M0120100J'] = integrate_Coss(dab['coss_C3M0120100J'])

    # Modulation Calculation
    # ZVS Modulation
    # calc_modulation(n, Ls, Lc1, Lc2, fs: np.ndarray | int | float, Coss1: np.ndarray, Coss2: np.ndarray,
    #                 V1: np.ndarray, V2: np.ndarray, P: np.ndarray)
    da_mod = mod_zvs.calc_modulation(dab.n,
                                     dab.Ls,
                                     dab.Lc1,
                                     dab.Lc2,
                                     dab.fs,
                                     dab.coss_C3M0120100J,
                                     dab.coss_C3M0120100J,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)

    # Unpack the results
    dab.append_result_dict(da_mod)

    debug(dab)

    # Plotting
    info("\nStart Plotting\n")
    v1_middle = int(np.shape(dab.mesh_P)[1] / 2)
    debug('View plane: U_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB()

    # Plot OptZVS mod results
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='OptZVS Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                        dab.mesh_P[:, v1_middle, :],
                        dab.mesh_V2[:, v1_middle, :],
                        dab.mod_zvs_phi[:, v1_middle, :],
                        dab.mod_zvs_tau1[:, v1_middle, :],
                        dab.mod_zvs_tau2[:, v1_middle, :],
                        mask1=dab.mod_zvs_mask_m1p[:, v1_middle, :],
                        mask2=dab.mod_zvs_mask_m2[:, v1_middle, :],
                        maskZVS=dab.mod_zvs_mask_zvs[:, v1_middle, :]
                        )

    # Plot all modulation angles but separately with autoscale
    plt.new_fig(nrows=1, ncols=3, tab_title='OptZVS Modulation Angles (autoscale)')
    plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                         dab.mesh_V2[:, v1_middle, :],
                         dab.mod_zvs_phi[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][0],
                         xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                         dab.mesh_V2[:, v1_middle, :],
                         dab.mod_zvs_tau1[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][1],
                         xlabel='P / W', ylabel='U2 / V', title='tau1 in rad')
    plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                         dab.mesh_V2[:, v1_middle, :],
                         dab.mod_zvs_tau2[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][2],
                         xlabel='P / W', ylabel='U2 / V', title='tau2 in rad')

    # Plot Coss
    plt.new_fig(nrows=1, ncols=2, tab_title='Coss C3M0120100J', sharex=False, sharey=False)
    plt.subplot(np.arange(dab['coss_C3M0120100J'].shape[0]),
                dab['coss_C3M0120100J'],
                ax=plt.figs_axes[-1][1][0],
                xlabel='U_DS / V', ylabel='C_oss / pF', title='Coss C3M0120100J',
                yscale='log')
    plt.subplot(np.arange(dab['qoss_C3M0120100J'].shape[0]),
                dab['qoss_C3M0120100J'],
                ax=plt.figs_axes[-1][1][1],
                xlabel='U_DS / V', ylabel='Q_oss / nC', title='Qoss C3M0120100J')

    plt.show()


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
    # ds.old_save_to_file(dab_specs, dab_results, name='mod_sps_sim_v21-v25-p19',
    #              comment='Simulation results for mod_sps with V1 10V res, V2 5V res and P 100W res.')


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
    ds.old_save_to_file(dab_specs, dab_results, directory='~/MA-LEA/LEA/Workdir/dab_optimizer_output', name='test-save',
                        comment='This is a saving test with random data!')
    # ds.old_save_to_file(dab_specs, dab_results, name='test-save', timestamp=False, comment='This is a saving test with random data!')

    # Loading
    # dab_specs_loaded, dab_results_loaded = ds.old_load_from_file('test-save.npz')
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
    dab_specs, dab_results = ds.old_load_from_file('~/MA-LEA/LEA/Workdir/dab_optimizer_output/test-sps-save.npz')
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
    dab_specs, dab_results = ds.old_load_from_file(dab_file)
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
    # dab_specs, dab_results = ds.old_load_from_file(dab_file)
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
    #     ds.old_save_to_csv(dab_specs, dab_results, key, directory, name)

    # sys.exit(0)
