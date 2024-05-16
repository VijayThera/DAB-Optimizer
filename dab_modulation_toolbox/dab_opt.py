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

import copy
import os
import pprint
import sys

import numpy as np
import math
# from datetime import datetime
# import logging
import argparse
# Status bar
from tqdm import tqdm

import dab_datasets as ds
from debug_tools import *
import debug_tools as db
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


@timeit
def plot_mod(dab, name, comment, directory, mod_keys, show_plot=True, logfile=str()):
    # Logging
    log = db.log(filename=os.path.join(directory, logfile) if logfile else '')
    ## Plotting
    log.info("\nStart Plotting\n")
    debug(mod_keys)

    # When dim=1 the v1_middle calc does not work.
    # Therefore, we stretch the array to use the same algo for every data.
    if dab.mesh_V1.shape[1] == 1:
        ## Broadcast arrays for plotting
        for k, v in dab.items():
            if k.startswith(('mesh_', 'mod_')):
                dab[k] = np.broadcast_to(dab[k], (dab.V2_step, 3, dab.P_step))

    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab.mesh_P)[1] / 2)
    log.info('View plane: U_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB(latex=False, show=show_plot, figsize=(15, 5), fontsize=22)

    for m in mod_keys:
        log.info('Plotting modulation: ' + m)

        # Set masks according to mod for later usage
        match m:
            case 'sps':
                mask1 = None
                mask2 = None
                mask3 = None
                maskZVS = None
            case 'mcl':
                mask1 = dab['mod_' + m + '_mask_tcm'][:, v1_middle, :]
                mask2 = dab['mod_' + m + '_mask_cpm'][:, v1_middle, :]
                mask3 = None
                maskZVS = None
            case s if s.startswith('zvs'):
                # Hide less useful masks
                mask1 = None
                mask2 = None
                # mask1 = dab['mod_' + m + '_mask_Im2'][:, v1_middle, :]
                # mask2 = dab['mod_' + m + '_mask_IIm2'][:, v1_middle, :]
                mask3 = dab['mod_' + m + '_mask_IIIm1'][:, v1_middle, :]
                maskZVS = dab['mod_' + m + '_mask_zvs'][:, v1_middle, :]
            case _:
                mask1 = None
                mask2 = None
                mask3 = None
                maskZVS = None

        # Plot all modulation angles
        plt.plot_modulation(dab.mesh_P[:, v1_middle, :],
                            dab.mesh_V2[:, v1_middle, :],
                            dab['mod_' + m + '_phi'][:, v1_middle, :],
                            dab['mod_' + m + '_tau1'][:, v1_middle, :],
                            dab['mod_' + m + '_tau2'][:, v1_middle, :],
                            mask1=mask1,
                            mask2=mask2,
                            mask3=mask3,
                            maskZVS=maskZVS,
                            tab_title=m + ' Modulation Angles'
                            )
        fname = 'mod_' + m + '_' + name + '_' + 'fig1'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

    info('Plotting is done!')
    # Finally show everything
    if show_plot:
        plt.show()
    else:
        plt.close()


@timeit
def plot_mod_sim(dab, name, comment, directory, mod_keys, show_plot=True, logfile=str()):
    # Logging
    log = db.log(filename=os.path.join(directory, logfile) if logfile else '')
    ## Plotting
    log.info("\nStart Plotting\n")
    debug(mod_keys)

    # When dim=1 the v1_middle calc does not work.
    # Therefore, we stretch the array to use the same algo for every data.
    if dab.mesh_V1.shape[1] == 1:
        ## Broadcast arrays for plotting
        for k, v in dab.items():
            if k.startswith(('mesh_', 'mod_', 'sim_')):
                dab[k] = np.broadcast_to(dab[k], (dab.V2_step, 3, dab.P_step))

    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab.mesh_P)[1] / 2)
    log.info('View plane: U_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0]))
    name += '_V1_{:.0f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB(latex=False, show=show_plot, figsize=(15, 5), fontsize=22)

    for m in mod_keys:
        log.info('Plotting modulation: ' + m)
        # Show ZVS coverage based on simulation:
        log.info(m + ' Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(
            round(np.array(dab['sim_' + m + '_zvs_coverage']).item(0), 3),
            round(np.array(dab['sim_' + m + '_zvs_coverage1']).item(0), 3),
            round(np.array(dab['sim_' + m + '_zvs_coverage2']).item(0), 3)))
        # Only non NaN areas:
        log.info(m + ' Simulation ZVS coverage (Bridge 1, Bridge 2) (non NaN): {} ({}, {})'.format(
            round(np.array(dab['sim_' + m + '_zvs_coverage_notnan']).item(0), 3),
            round(np.array(dab['sim_' + m + '_zvs_coverage1_notnan']).item(0), 3),
            round(np.array(dab['sim_' + m + '_zvs_coverage2_notnan']).item(0), 3)))
        # Mean of I1:
        log.info(m + ' Simulation I_1-total-mean: {}'.format(
            round(np.array(dab['sim_' + m + '_i_HF1_total_mean']).item(0), 3)))
        log.info(m + ' Simulation I^2_1-total-mean: {}'.format(
            round(np.array(dab['sim_' + m + '_I1_squared_total_mean']).item(0), 3)))

        # Set masks according to mod for later usage
        match m:
            case 'sps':
                mask1 = None
                mask2 = None
                mask3 = None
                maskZVS = None
            case 'mcl':
                mask1 = dab['mod_' + m + '_mask_tcm'][:, v1_middle, :]
                mask2 = dab['mod_' + m + '_mask_cpm'][:, v1_middle, :]
                mask3 = None
                maskZVS = None
            case s if s.startswith('zvs'):
                # Hide less useful masks
                mask1 = None
                mask2 = None
                # mask1 = dab['mod_' + m + '_mask_Im2'][:, v1_middle, :]
                # mask2 = dab['mod_' + m + '_mask_IIm2'][:, v1_middle, :]
                mask3 = dab['mod_' + m + '_mask_IIIm1'][:, v1_middle, :]
                maskZVS = dab['mod_' + m + '_mask_zvs'][:, v1_middle, :]
            case _:
                mask1 = None
                mask2 = None
                mask3 = None
                maskZVS = None

        # Plot all modulation angles
        plt.plot_modulation(dab.mesh_P[:, v1_middle, :],
                            dab.mesh_V2[:, v1_middle, :],
                            dab['mod_' + m + '_phi'][:, v1_middle, :],
                            dab['mod_' + m + '_tau1'][:, v1_middle, :],
                            dab['mod_' + m + '_tau2'][:, v1_middle, :],
                            mask1=mask1,
                            mask2=mask2,
                            mask3=mask3,
                            maskZVS=maskZVS,
                            tab_title=m + ' Modulation Angles'
                            )
        fname = 'mod_' + m + '_' + name + '_' + 'fig1'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # v_ds plots
        plt.new_fig(nrows=1, ncols=2, tab_title=m + ' ZVS')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_v_ds_S11_sw_on'][:, v1_middle, :] / dab.mesh_V1[:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             z_min=0,
                             z_max=1,
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$u_\mathrm{DS,S11,sw-on} \:/\: U_\mathrm{DC1}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_v_ds_S23_sw_on'][:, v1_middle, :] / dab.mesh_V2[:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             z_min=0,
                             z_max=1,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$u_\mathrm{DS,S23,sw-on} \:/\: U_\mathrm{DC2}$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig2'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # i_l plots 'i_HF1_S11_sw_on', 'i_HF2_S23_sw_on'
        plt.new_fig(nrows=1, ncols=2, tab_title=m + ' i_L')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_i_HF1_S11_sw_on'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$i_\mathrm{1,S11,sw-on} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_i_HF2_S23_sw_on'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$i_\mathrm{2,S23,sw-on} \:/\: \mathrm{A}$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig3'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # Total loss
        plt.new_fig(nrows=1, ncols=3, tab_title=m + ' Total Loss')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_power_deviation'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][0],
                             z_min=0.5,
                             z_max=1.5,
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$P_\mathrm{out,Sim} \:/\: P_\mathrm{out,desired}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_p_sw1'][:, v1_middle, :] + dab['sim_' + m + '_p_sw2'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{sw,total} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_p_cond1'][:, v1_middle, :]
                             + dab['sim_' + m + '_p_cond2'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{cond,total} \:/\: \mathrm{W}$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig4'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # Plot power loss
        plt.new_fig(nrows=1, ncols=4, tab_title=m + ' Power Loss')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_S11_p_sw'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$', ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$P_\mathrm{S11,sw} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_S11_p_cond'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{S11,cond} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_S23_p_sw'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{S23,sw} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_S23_p_cond'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][3],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{S23,cond} \:/\: \mathrm{W}$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig5'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # Plot inductor currents
        plt.new_fig(nrows=1, ncols=3, tab_title=m + ' Inductor currents')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_i_HF1'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$I_\mathrm{1} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_i_Lc1'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$I_\mathrm{L1} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                             dab.mesh_V2[:, v1_middle, :],
                             dab['sim_' + m + '_i_Lc2'][:, v1_middle, :] / dab.n,
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$I^\prime_\mathrm{L2} \:/\: \mathrm{A}$')
        fname = 'mod_' + m + '_' + name + '_' + 'fig6'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)
        if not show_plot:
            plt.close()

    info('Plotting is done!')
    # Finally show everything
    if show_plot:
        plt.show()
    else:
        plt.close()


@timeit
def dab_mod_save():
    """
    Run the modulation optimization procedure and save the results in a file
    """
    ## Set the basic DAB Specification
    dab = ds.dab_ds_default_Gv7()
    dab.n = 4

    dab.V2_step = 50
    dab.P_step = 38
    dab.P_min = -2200

    dab.Ls = 85e-6
    dab.Lc1 = 800e-6
    dab.Lc2 = 611e-6 / (dab.n ** 2)
    # Generate meshes
    dab.gen_meshes()

    # Coss + Cpar
    dab.coss_1 = (dab.coss_1 + dab.C_HB11)
    dab.coss_2 = (dab.coss_2 + dab.C_HB22)

    # Set file names
    directory = '../dab_modulation_output/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    name_extra = 'original-setup'
    name_L = 'Ls_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(dab.Ls * 1e6),
                                                  int(dab.Lc1 * 1e6),
                                                  int(dab.Lc2 * 1e6))
    name = 'mod_Gv8_' + name_extra + '_' + name_L + '_v{}-v{}-p{}'.format(int(dab.V1_step),
                                                                          int(dab.V2_step),
                                                                          int(dab.P_step))
    name_pre = 'mod_sps_mcl_zvs_'
    if __debug__:
        name_pre = 'debug_' + name_pre
    comment = 'Only modulation results for mod_sps, mod_mcl and mod_zvs with V1 {}, V2 {} and P {} steps.'.format(
        int(dab.V1_step),
        int(dab.V2_step),
        int(dab.P_step))
    if __debug__:
        comment = 'Debug ' + comment

    ## Saving
    # Create new dir for all files
    directory = directory + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name_pre + name
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Logging
    log = db.log(filename=os.path.join(directory, 'dab_opt.log'))

    ## Modulation Calculation
    # SPS Modulation
    log.info('Starting SPS Modulation...')
    da_mod = mod_sps.calc_modulation(dab.n,
                                     dab.Ls,
                                     dab.fs,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_sps_')

    ## Modulation Calculation
    # MCL Modulation
    log.info('Starting MCL Modulation...')
    da_mod = mod_mcl.calc_modulation(dab.n,
                                     dab.Ls,
                                     dab.fs,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_mcl_')

    ## Modulation Calculation
    # ZVS Modulation
    log.info('Starting ZVS Modulation...')
    da_mod = mod_zvs.calc_modulation(dab.n,
                                     dab.Ls,
                                     dab.Lc1,
                                     dab.Lc2,
                                     dab.fs,
                                     dab.coss_1,
                                     dab.coss_2,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')

    # Save data
    # ds.save_to_file(dab, directory=directory, name=name, comment=comment)
    dab.save_to_file(directory=directory, name='dab_' + name, comment=comment, timestamp=False)
    dab.pprint_to_file(os.path.join(directory, 'dab_' + name + '.txt'))

    # Save to csv for DAB-Controller
    # Meshes to save:
    keys = ['mod_sps_phi', 'mod_sps_tau1', 'mod_sps_tau2',
            'mod_mcl_phi', 'mod_mcl_tau1', 'mod_mcl_tau2',
            'mod_zvs_phi', 'mod_zvs_tau1', 'mod_zvs_tau2']
    # Convert phi, tau1/2 from rad to duty cycle * 10000
    # In DAB-Controller we need duty cycle * 10000 (2pi eq. 10000)
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 10000
        ds.save_to_csv(dab2, key, directory, 'control_' + name, timestamp=False)
    # Convert phi, tau1/2 from rad to degree
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 360
        ds.save_to_csv(dab2, key, directory, 'degree_' + name, timestamp=False)

    ## Plotting
    # plot_mod_sim(dab, name, comment, directory, ['sps', 'mcl', 'zvs', 'zvs2'])
    # FIXME do only mod plot here!
    plot_mod(dab, name, comment, directory, ['sps', 'mcl', 'zvs'])


@timeit
def dab_sim_save():
    """
    Run the complete optimization procedure and save the results in a file
    """
    ## Set the basic DAB Specification
    dab = ds.dab_ds_default_Gv7()
    # dab.V1_min = 700
    # dab.V1_max = 700
    # dab.V1_step = 1
    # dab.V2_step = 13
    # dab.P_step = 10
    # dab.V2_step = 25
    # dab.P_step = 19
    # dab.P_min = 0
    # For Measurements
    # dab.V2_step = 7
    # dab.P_step = 7
    # dab.P_min = 400
    dab.V2_step = 50
    dab.P_step = 38
    # For mod_zvs n_min requirement
    # dab.n = 4
    # Iter results
    dab.Ls = 85e-6
    dab.Lc1 = 800e-6
    dab.Lc2 = 611e-6 / (dab.n ** 2)
    dab.t_dead1 = 150e-9
    dab.t_dead2 = 150e-9
    # Generate meshes
    dab.gen_meshes()

    # Coss + Cpar
    dab.coss_1 = (dab.coss_1 + dab.C_HB11)
    dab.coss_2 = (dab.coss_2 + dab.C_HB22)

    ## Set sim defaults
    simfilepath = '../circuits/DAB_MOSFET_Modulation_v8.ipes'
    timestep = 1e-9
    simtime = 50e-6
    timestep_pre = 50e-9
    simtime_pre = 5e-3
    # geckoport = 43036
    # Automatically select a free port
    geckoport = 0
    # Set file names
    directory = '../dab_modulation_output/'
    name_extra = 'orig-with-Lc1-tdead150'
    name_L = 'Ls_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(dab.Ls * 1e6),
                                                  int(dab.Lc1 * 1e6),
                                                  int(dab.Lc2 * 1e6))
    name = 'sim_Gv8_' + name_extra + '_' + name_L + '_v{}-v{}-p{}'.format(int(dab.V1_step),
                                                                          int(dab.V2_step),
                                                                          int(dab.P_step))
    name_pre = 'mod_sps_mcl_zvs_'
    if __debug__:
        name_pre = 'debug_' + name_pre
    comment = 'Simulation results for mod_sps, mod_mcl and mod_zvs with V1 {}, V2 {} and P {} steps.'.format(
        int(dab.V1_step),
        int(dab.V2_step),
        int(dab.P_step))
    comment = comment + '\n' + 'Using simfilepath = ' + simfilepath
    comment = comment + '\n' + 'Sim for original setup as baseline'
    if __debug__:
        comment = 'Debug ' + comment

    ## Saving
    # Create new dir for all files
    directory = directory + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name_pre + name
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Logging
    log = db.log(filename=os.path.join(directory, 'dab_opt.log'))

    ## Modulation Calculation
    # SPS Modulation
    log.info('Starting SPS Modulation...')
    da_mod = mod_sps.calc_modulation(dab.n,
                                     dab.Ls,
                                     dab.fs,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_sps_')
    # Simulation
    da_sim = sim_gecko.start_sim(dab.mesh_V1,
                                 dab.mesh_V2,
                                 dab.mesh_P,
                                 dab.mod_sps_phi,
                                 dab.mod_sps_tau1,
                                 dab.mod_sps_tau2,
                                 dab.t_dead1,
                                 dab.t_dead2,
                                 dab.fs,
                                 dab.Ls,
                                 dab.Lc1,
                                 dab.Lc2,
                                 dab.n,
                                 dab.temp,
                                 simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport=geckoport)
    # Unpack the results
    dab.append_result_dict(da_sim, name_pre='sim_sps_')

    # Show ZVS coverage based on simulation
    log.info('Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(dab.sim_sps_zvs_coverage,
                                                                                dab.sim_sps_zvs_coverage1,
                                                                                dab.sim_sps_zvs_coverage2))
    log.info('Simulation I_1-total-mean: {}'.format(dab.sim_sps_i_HF1_total_mean))

    ## Modulation Calculation
    # MCL Modulation
    log.info('Starting MCL Modulation...')
    da_mod = mod_mcl.calc_modulation(dab.n,
                                     dab.Ls,
                                     dab.fs,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_mcl_')
    # Simulation
    da_sim = sim_gecko.start_sim(dab.mesh_V1,
                                 dab.mesh_V2,
                                 dab.mesh_P,
                                 dab.mod_mcl_phi,
                                 dab.mod_mcl_tau1,
                                 dab.mod_mcl_tau2,
                                 dab.t_dead1,
                                 dab.t_dead2,
                                 dab.fs,
                                 dab.Ls,
                                 dab.Lc1,
                                 dab.Lc2,
                                 dab.n,
                                 dab.temp,
                                 simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport=geckoport)
    # Unpack the results
    dab.append_result_dict(da_sim, name_pre='sim_mcl_')

    log.info('Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(dab.sim_mcl_zvs_coverage,
                                                                                dab.sim_mcl_zvs_coverage1,
                                                                                dab.sim_mcl_zvs_coverage2))
    log.info('Simulation I_1-total-mean: {}'.format(dab.sim_mcl_i_HF1_total_mean))

    ## Modulation Calculation
    # ZVS Modulation
    log.info('Starting ZVS Modulation...')
    da_mod = mod_zvs.calc_modulation(dab.n,
                                     dab.Ls,
                                     dab.Lc1,
                                     dab.Lc2,
                                     dab.fs,
                                     dab.coss_1,
                                     dab.coss_2,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    # Simulation
    da_sim = sim_gecko.start_sim(dab.mesh_V1,
                                 dab.mesh_V2,
                                 dab.mesh_P,
                                 dab.mod_zvs_phi,
                                 dab.mod_zvs_tau1,
                                 dab.mod_zvs_tau2,
                                 dab.t_dead1,
                                 dab.t_dead2,
                                 dab.fs,
                                 dab.Ls,
                                 dab.Lc1,
                                 dab.Lc2,
                                 dab.n,
                                 dab.temp,
                                 simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport=geckoport)
    # Unpack the results
    dab.append_result_dict(da_sim, name_pre='sim_zvs_')

    ## Show ZVS coverage based on calculation
    log.info('Calculation ZVS coverage: {}'.format(dab.mod_zvs_zvs_coverage))
    # Show ZVS coverage based on simulation
    log.info('Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(dab.sim_zvs_zvs_coverage,
                                                                                dab.sim_zvs_zvs_coverage1,
                                                                                dab.sim_zvs_zvs_coverage2))

    ## Only non NaN areas:
    ## Show ZVS coverage based on calculation
    log.info('Calculation ZVS coverage (non NaN): {}'.format(dab.mod_zvs_zvs_coverage_notnan))
    # Show ZVS coverage based on simulation
    log.info('Simulation ZVS coverage (Bridge 1, Bridge 2) (non NaN): {} ({}, {})'.format(
        dab.sim_zvs_zvs_coverage_notnan, dab.sim_zvs_zvs_coverage1_notnan,
        dab.sim_zvs_zvs_coverage2_notnan))
    log.info('Simulation I_1-total-mean: {}'.format(dab.sim_zvs_i_HF1_total_mean))

    # Save data
    # ds.save_to_file(dab, directory=directory, name=name, comment=comment)
    dab.save_to_file(directory=directory, name='dab_' + name, comment=comment, timestamp=False)
    dab.pprint_to_file(os.path.join(directory, 'dab_' + name + '.txt'))

    # Save to csv for DAB-Controller
    # Meshes to save:
    keys = ['mod_sps_phi', 'mod_sps_tau1', 'mod_sps_tau2',
            'mod_mcl_phi', 'mod_mcl_tau1', 'mod_mcl_tau2',
            'mod_zvs_phi', 'mod_zvs_tau1', 'mod_zvs_tau2']
    # Convert phi, tau1/2 from rad to duty cycle * 10000
    # In DAB-Controller we need duty cycle * 10000 (2pi eq. 10000)
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 10000
        ds.save_to_csv(dab2, key, directory, 'control_' + name, timestamp=False)
    # Convert phi, tau1/2 from rad to degree
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 360
        ds.save_to_csv(dab2, key, directory, 'degree_' + name, timestamp=False)

    ## Plotting
    # plot_mod_sim(dab, name, comment, directory, ['sps', 'mcl', 'zvs', 'zvs2'])
    plot_mod_sim(dab, name, comment, directory, ['sps', 'mcl', 'zvs'])


def dab_sim_save_zvs():
    """
    Run the complete optimization procedure and save the results in a file
    """
    ## Set the basic DAB Specification
    dab = ds.dab_ds_default_Gv8_sim()
    dab.t_dead1 = 100e-9
    dab.t_dead2 = 100e-9
    # dab = ds.dab_ds_default_Gv7()
    # dab.V1_min = 700
    # dab.V1_max = 700
    # dab.V1_step = 1
    # # dab.V2_step = 13
    # # dab.P_step = 10
    # dab.V2_step = 25
    # dab.P_step = 19
    # dab.P_min = 0
    # # For Measurements
    # dab.V2_step = 7
    # dab.P_step = 7
    # dab.P_min = 400
    # # For mod_zvs n_min requirement
    # dab.n = 4
    # # dab.Ls = 110e-6
    # # dab.Lc1 = 25.62e-3
    # # dab.Lc2 = 611e-6
    # # dab.Lc1 = 1.25e-3
    # dab.Lc2 = 1.25e-3 / (dab.n ** 2)
    # dab.Lc1 = 600e-6
    # # dab.Lc2 = 600e-6 / (dab.n ** 2)
    # dab.C_HB11 = 32e-12
    # dab.t_dead1 = 250e-9
    # # dab.t_dead2 = 50e-9
    # # # Assumption for tests
    # # dab.Ls = 85e-6
    # # dab.Lc1 = 25.62e-3
    # # dab.Lc2 = 611e-6
    # # dab.C_PCB_leg = 5e-12
    # # # testing...
    # # # Iter results
    # # dab.Ls = 85e-6
    # # dab.Lc1 = 1000e-6
    # # dab.Lc2 = 1000e-6 / (dab.n ** 2)
    # # dab.C_PCB_leg = 1e-15
    # # Generate meshes
    # dab.gen_meshes()

    dab.coss_1 = dab.coss_1 + dab.C_HB11
    dab.coss_2 = dab.coss_2 + dab.C_HB22

    ## Set sim defaults
    simfilepath = '../circuits/DAB_MOSFET_Modulation_v8.ipes'
    timestep = 1e-9
    simtime = 50e-6
    timestep_pre = 50e-9
    simtime_pre = 5e-3
    # geckoport = 43036
    # Automatically select a free port
    geckoport = 0
    # Set file names
    directory = '../dab_modulation_output/'
    name_extra = 'n3_tdead1_100_Coss2'
    name_L = 'Ls_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(dab.Ls * 1e6),
                                                  int(dab.Lc1 * 1e6),
                                                  int(dab.Lc2 * 1e6))
    name = 'sim_Gv8_' + name_extra + '_' + name_L + '_v{}-v{}-p{}'.format(int(dab.V1_step),
                                                                          int(dab.V2_step),
                                                                          int(dab.P_step))
    name_pre = 'mod_zvs_'
    if __debug__:
        name_pre = 'debug_' + name_pre
    comment = 'Simulation results for mod_zvs with V1 {}, V2 {} and P {} steps.'.format(
        int(dab.V1_step),
        int(dab.V2_step),
        int(dab.P_step))
    comment = comment + '\n' + 'Using simfilepath = ' + simfilepath
    if __debug__:
        comment = 'Debug ' + comment

    ## Saving
    # Create new dir for all files
    directory = directory + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name_pre + name
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Logging
    log = db.log(filename=os.path.join(directory, 'dab_opt.log'))

    ## Modulation Calculation
    # ZVS Modulation
    log.info('Starting ZVS Modulation...')
    da_mod = mod_zvs.calc_modulation(dab.n,
                                     dab.Ls,
                                     dab.Lc1,
                                     dab.Lc2,
                                     dab.fs,
                                     dab.coss_1,
                                     dab.coss_2,
                                     dab.mesh_V1,
                                     dab.mesh_V2,
                                     dab.mesh_P)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    # Simulation
    da_sim = sim_gecko.start_sim(dab.mesh_V1,
                                 dab.mesh_V2,
                                 dab.mesh_P,
                                 dab.mod_zvs_phi,
                                 dab.mod_zvs_tau1,
                                 dab.mod_zvs_tau2,
                                 dab.t_dead1,
                                 dab.t_dead2,
                                 dab.fs,
                                 dab.Ls,
                                 dab.Lc1,
                                 dab.Lc2,
                                 dab.n,
                                 dab.temp,
                                 simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport=geckoport)
    # Unpack the results
    dab.append_result_dict(da_sim, name_pre='sim_zvs_')

    ## Show ZVS coverage based on calculation
    log.info('Calculation ZVS coverage: {}'.format(dab.mod_zvs_zvs_coverage))
    # Show ZVS coverage based on simulation
    log.info('Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(dab.sim_zvs_zvs_coverage,
                                                                                dab.sim_zvs_zvs_coverage1,
                                                                                dab.sim_zvs_zvs_coverage2))

    ## Only non NaN areas:
    ## Show ZVS coverage based on calculation
    log.info('Calculation ZVS coverage (non NaN): {}'.format(dab.mod_zvs_zvs_coverage_notnan))
    # Show ZVS coverage based on simulation
    log.info('Simulation ZVS coverage (Bridge 1, Bridge 2) (non NaN): {} ({}, {})'.format(
        dab.sim_zvs_zvs_coverage_notnan, dab.sim_zvs_zvs_coverage1_notnan,
        dab.sim_zvs_zvs_coverage2_notnan))

    # Save data
    # ds.save_to_file(dab, directory=directory, name=name, comment=comment)
    dab.save_to_file(directory=directory, name='dab_' + name, comment=comment, timestamp=False)
    dab.pprint_to_file(os.path.join(directory, 'dab_' + name + '.txt'))

    # Save to csv for DAB-Controller
    # Meshes to save:
    keys = ['mod_zvs_phi', 'mod_zvs_tau1', 'mod_zvs_tau2']
    # Convert phi, tau1/2 from rad to duty cycle * 10000
    # In DAB-Controller we need duty cycle * 10000 (2pi eq. 10000)
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 10000
        ds.save_to_csv(dab2, key, directory, 'control_' + name, timestamp=False)
    # Convert phi, tau1/2 from rad to degree
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 360
        ds.save_to_csv(dab2, key, directory, 'degree_' + name, timestamp=False)

    ## Plotting
    plot_mod_sim(dab, name, comment, directory, ['zvs'])


@timeit
def dab_iterate_2D(sim=False, save_all=False, zvs_plot_limit=1, iter='Lc'):
    """
    Run the complete optimization procedure and save the results in a file
    """
    ## Set the basic DAB Specification
    dab = ds.dab_ds_default_Gv8_sim()
    # Assuming theoretical transformer
    # dab = ds.dab_ds_default_Gv8_sim_n4()
    # Assumption for tests
    if iter == 'Lc':
        dab.t_dead1 = 150e-9
        dab.t_dead2 = 150e-9
    if iter == 'Ls':
        dab.t_dead1 = 150e-9
        dab.t_dead2 = 150e-9
    if iter == 'tdead':
        # Assume some sane Lc values
        dab.Lc1 = 750e-6
        dab.Lc2 = 611e-6 / (dab.n ** 2)
        # dab.Lc2 = 55e-9

    # For Sim
    if sim:
        dab.V2_step = 7
        dab.P_step = 7
    # Generate meshes
    dab.gen_meshes()

    # Overestimate Coss and Cpar
    dab.coss_1 = dab.coss_1 + dab.C_HB11
    dab.coss_2 = dab.coss_2 + dab.C_HB22

    # Iteration params
    # zvs_plot_limit = 0.9
    steps = 50
    # testing...
    if sim:
        steps = 12  # ~200min
        # steps = 2  # for fast test

    if iter == 'Lc':
        # Lc1
        x_min = 400e-6
        x_max = 1400e-6
        # Lc2
        y_min = 400e-6
        y_max = 1400e-6
        # mark line plot
        x_line = (400, 1400)
        y_line = (611, 611)
        # mark star
        x_star = 800
        y_star = 611
    if iter == 'Ls':
        # Ls
        x_min = 60e-6
        x_max = 120e-6
        # Lc1
        y_min = 400e-6
        y_max = 1400e-6
        # mark line plot
        x_line = (85, 85)
        y_line = (400, 1400)
        # mark star
        x_star = None
        y_star = None
    if iter == 'tdead':
        # tdead1
        x_min = 50e-9
        x_max = 200e-9
        # tdead2
        y_min = 50e-9
        y_max = 200e-9
        steps = 7

    # if iter == 'Lc':
    # if iter == 'Ls':
    # if iter == 'tdead':

    # Set file names
    directory = '../dab_modulation_output/'

    # Set names in case we want to save something
    name_pre = 'mod_zvs_'

    if iter == 'Lc':
        name = 'Iter_Lc1{}-{}uH_Lc2{}-{}uH'.format(int(x_min * 1e6), int(x_max * 1e6), int(y_min * 1e6),
                                                   int(y_max * 1e6))
    if iter == 'Ls':
        name = 'Iter_Ls{}-{}uH_Lc12_{}-{}uH'.format(int(x_min * 1e6), int(x_max * 1e6), int(y_min * 1e6),
                                                    int(y_max * 1e6))
    if iter == 'tdead':
        name = 'Iter_tdead1_{}-{}ns_tdead2_{}-{}ns'.format(int(x_min * 1e9), int(x_max * 1e9), int(y_min * 1e9),
                                                           int(y_max * 1e9))

    comment = 'Iteration of ' + name_pre + name
    fname = name_pre + name + '_ZVS-Coverage'

    # Create new dir for all files
    directory = directory + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name_pre + name
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    if not __debug__:
        if not os.path.exists(directory):
            os.mkdir(directory)

    ## Set sim defaults
    simfilepath = '../circuits/DAB_MOSFET_Modulation_v8.ipes'
    timestep = 1e-9
    simtime = 50e-6
    timestep_pre = 50e-9
    simtime_pre = 5e-3
    # geckoport = 43036
    # Automatically select a free port
    geckoport = 0
    geckoport = sim_gecko.get_free_port(43192, 43256)

    mod_keys = []

    # Create result meshes
    results = ds.DAB_Data()
    # Create meshgrid for iter values
    if iter == 'Lc':
        results.mesh_x, results.mesh_y = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
        results.mesh_x_name = 'Lc1'
        results.mesh_y_name = 'Lc2'
    if iter == 'Ls':
        results.mesh_x, results.mesh_y = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
        results.mesh_x_name = 'Ls'
        results.mesh_y_name = 'Lc1'
    if iter == 'tdead':
        results.mesh_x, results.mesh_y = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
        results.mesh_x_name = 't_dead1'
        results.mesh_y_name = 't_dead2'

    sim_keys = ['p_sw1', 'p_cond1', 'p_sw2', 'p_cond2', 'v_ds_S11_sw_on', 'v_ds_S23_sw_on', 'i_HF1_S11_sw_on',
                'i_HF2_S23_sw_on']
    for key in sim_keys:
        results['iter_sim_' + key] = np.full_like(results.mesh_x, np.nan)
    # result_keys = ['iter_zvs_coverage_mod', 'iter_zvs_coverage_sim', 'iter_i_HF1_total_mean_sim', 'iter_', 'iter_', 'iter_', 'iter_', 'iter_', 'iter_', 'iter_']
    results.iter_mod_zvs_coverage = np.full_like(results.mesh_x, np.nan)
    results.iter_mod_contains_nan = np.full_like(results.mesh_x, np.nan)
    results.iter_sim_zvs_coverage = np.full_like(results.mesh_x, np.nan)
    results.iter_sim_i_HF1_total_mean = np.full_like(results.mesh_x, np.nan)
    results.iter_sim_I1_squared_total_mean = np.full_like(results.mesh_x, np.nan)

    if sim and not __debug__:
        # Logging
        log = db.log(filename=os.path.join(directory, 'dab_opt.log'))
        log.info('Starting L iteration: ' + name)
    else:
        info('Starting L iteration: ' + name)

    # Progressbar init
    # Calc total number of iterations to simulate
    it_total = results.mesh_x.size
    pbar = tqdm(total=it_total)

    # for Ls in np.linspace(Ls_min, Ls_max, steps):
    #     for Lc in np.linspace(Lc_min, Lc_max, steps):
    for vec in np.ndindex(results.mesh_x.shape):
        if iter == 'Lc':
            # Set Lc1 and Lc2 according to assumption
            Ls = dab.Ls
            Lc1 = results.mesh_x[vec]
            Lc2 = results.mesh_y[vec] / (dab.n ** 2)
            t_dead1 = dab.t_dead1
            t_dead2 = dab.t_dead2
            # Set names in case we want to save something
            _name = 'Ls_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(Ls * 1e6),
                                                         int(Lc1 * 1e6),
                                                         int(Lc2 * 1e6))
        if iter == 'Ls':
            # Set Ls and Lc1/2 according to assumption
            Ls = results.mesh_x[vec]
            Lc1 = results.mesh_y[vec]
            Lc2 = results.mesh_y[vec] / (dab.n ** 2)
            t_dead1 = dab.t_dead1
            t_dead2 = dab.t_dead2
            # Set names in case we want to save something
            _name = 'Ls_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(Ls * 1e6),
                                                         int(Lc1 * 1e6),
                                                         int(Lc2 * 1e6))
        if iter == 'tdead':
            # Set tdead1/2 according to assumption
            Ls = dab.Ls
            Lc1 = dab.Lc1
            Lc2 = dab.Lc2
            t_dead1 = round(results.mesh_x[vec], 9)
            t_dead2 = round(results.mesh_y[vec], 9)
            # Set names in case we want to save something
            _name = 'tdead1_{}ns__tdead2_{}ns'.format(int(t_dead1 * 1e9),
                                                      int(t_dead2 * 1e9))
            # Important to have timestep always smaller than any time var in simulation model
            if (t_dead1 < timestep_pre) or (t_dead2 < timestep_pre):
                timestep_pre = t_dead1 if t_dead1 < t_dead2 else t_dead2
            info(timestep_pre, t_dead1, t_dead2)

        ## Modulation Calculation
        # ZVS Modulation
        # info('Starting ZVS Modulation... L_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(Ls * 1e6), int(Lc1 * 1e6),
        #                                                                   int(Lc2 * 1e6)))
        da_mod = mod_zvs.calc_modulation(dab.n,
                                         Ls,
                                         Lc1,
                                         Lc2,
                                         dab.fs,
                                         dab.coss_1,
                                         dab.coss_2,
                                         dab.mesh_V1,
                                         dab.mesh_V2,
                                         dab.mesh_P)

        # Save coverage for plot
        results.iter_mod_zvs_coverage[vec] = da_mod['zvs_coverage']
        # results.iter_mod_contains_nan[vec] = np.isnan(da_mod['tau1']).any()
        results.iter_mod_contains_nan[vec] = np.mean(np.isnan(da_mod['tau1']))

        if sim:
            # Simulation
            da_sim = sim_gecko.start_sim(dab.mesh_V1,
                                         dab.mesh_V2,
                                         dab.mesh_P,
                                         da_mod['phi'],
                                         da_mod['tau1'],
                                         da_mod['tau2'],
                                         t_dead1,
                                         t_dead2,
                                         dab.fs,
                                         Ls,
                                         Lc1,
                                         Lc2,
                                         dab.n,
                                         dab.temp,
                                         simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport=geckoport)

            # Save coverage for plot
            results.iter_sim_zvs_coverage[vec] = da_sim['zvs_coverage_notnan']
            results.iter_sim_i_HF1_total_mean[vec] = da_sim['i_HF1_total_mean']
            results.iter_sim_I1_squared_total_mean[vec] = da_sim['I1_squared_total_mean']
            for key in sim_keys:
                results['iter_sim_' + key][vec] = np.nanmean(da_sim[key])

            if ((results.iter_sim_zvs_coverage[vec] >= zvs_plot_limit) or save_all) and not __debug__:
                log.info('\n' + _name)
                log.info('Calculation ZVS coverage: {}'.format(da_mod['zvs_coverage']))
                log.info('Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(da_sim['zvs_coverage'],
                                                                                            da_sim['zvs_coverage1'],
                                                                                            da_sim['zvs_coverage2']))
                log.info('Simulation I_1-total-mean: {}'.format(da_sim['i_HF1_total_mean']))
                log.info('Simulation I^2_1-total-mean: {}'.format(da_sim['I1_squared_total_mean']))
                # Unpack the results
                dab.append_result_dict(da_mod, name_pre='mod_zvs_' + _name + '_')
                dab.append_result_dict(da_sim, name_pre='sim_zvs_' + _name + '_')

            if (results.iter_sim_zvs_coverage[vec] >= zvs_plot_limit) and not __debug__:
                ## remember for Plotting
                mod_keys.append('zvs_' + _name)

        # Progressbar update, default increment +1
        pbar.update()

    # Progressbar end
    pbar.close()

    if not __debug__:
        # numpy saves everything for us in a handy zip file
        results.save_to_file(directory, name_pre + name, False, comment)
        results.pprint_to_file(os.path.join(directory, name_pre + name + 'pprint.txt'))
        # Readable DAB Specs except right Ls,Lc1,Lc2
        dab.save_to_file(directory=directory, name='dab_' + name, comment=name, timestamp=False)
        dab.pprint_to_file(os.path.join(directory, 'dab_' + name + '_pprint.txt'))

    if sim:
        ## Plotting
        # FIXME This may take a long time
        info('Start plotting dab for selected ZVS areas. This may take some time.')
        plot_mod_sim(dab, name, comment, directory, mod_keys, show_plot=False)

    # Find a common z_min
    z_min = round(float(min(np.nanmin(results.iter_mod_zvs_coverage), np.nanmin(results.iter_sim_zvs_coverage))),
                  ndigits=1)
    debug('z_min: {}'.format(z_min))
    debug('z_max mod, sim: {}, {}'.format(np.nanmax(results.iter_mod_zvs_coverage),
                                          np.nanmax(results.iter_sim_zvs_coverage)))

    if iter == 'Lc':
        x = results.mesh_x * 1e6
        y = results.mesh_y * 1e6
        xlabel = r'$L_\mathrm{1} \:/\: \mathrm{\mu H}$'
        ylabel = r'$L^\prime_\mathrm{2} \:/\: \mathrm{\mu H}$'
        same_xy_ticks = True
    else:
        same_xy_ticks = False
    if iter == 'Ls':
        x = results.mesh_x * 1e6
        y = results.mesh_y * 1e6
        xlabel = r'$L \:/\: \mathrm{\mu H}$'
        ylabel = r'$L_\mathrm{1},\, L^\prime_\mathrm{2} \:/\: \mathrm{\mu H}$'
    if iter == 'tdead':
        x = results.mesh_x * 1e9
        y = results.mesh_y * 1e9
        xlabel = r'$t_\mathrm{dead1} \:/\: \mathrm{ns}$'
        ylabel = r'$t_\mathrm{dead2} \:/\: \mathrm{ns}$'

    ## Plotting the zvs coverage
    plt = plot_dab.Plot_DAB(latex=False, figsize=(5, 4))
    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS Coverage')
    plt.subplot_contourf(x,
                         y,
                         results.iter_mod_zvs_coverage,
                         z_min=z_min,
                         z_max=1,
                         num_cont_lines=12,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'ZVS Abdeckung',
                         same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_mod', comment, timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS Coverage Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_zvs_coverage,
                         z_min=z_min,
                         z_max=1,
                         num_cont_lines=12,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'ZVS Abdeckung',
                         same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_sim', comment, timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='I^2_1 total mean Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_I1_squared_total_mean,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{I^2_1} \:/\: \mathrm{A}$',
                         same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_I1_squared_total_mean_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF1 total mean Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_i_HF1_total_mean,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{I}_\mathrm{1} \:/\: \mathrm{A}$',
                         same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF1_total_mean_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF1_S11_sw_on mean Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_i_HF1_S11_sw_on,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{i}_\mathrm{1,S11,sw-on} \:/\: \mathrm{A}$',
                         same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF1_S11_sw_on_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF2_S23_sw_on mean Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_i_HF2_S23_sw_on,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{i}_\mathrm{2,S23,sw-on} \:/\: \mathrm{A}$',
                         same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF2_S23_sw_on_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='sw loss')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_p_sw1 + results.iter_sim_p_sw2,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{P}_\mathrm{sw-loss} \:/\: \mathrm{W}$',
                         same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_p-sw_sim', comment, timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='cond loss Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_p_cond1 + results.iter_sim_p_cond2,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{P}_\mathrm{cond-loss} \:/\: \mathrm{W}$',
                         same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_p-cond_sim', comment, timestamp=False)

    if not (results.iter_mod_contains_nan == results.iter_mod_contains_nan[0]).all():
        plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS NAN')
        plt.subplot_contourf(x,
                             y,
                             results.iter_mod_contains_nan,
                             # results.iter_mod_zvs_coverage,
                             # nan_matrix=nanmatrix,
                             ax=plt.figs_axes[-1][1],
                             xlabel=xlabel,
                             ylabel=ylabel,
                             title=r'NAN Abdeckung',
                             same_xy_ticks=same_xy_ticks)
    if x_line is not None:
        plt.figs_axes[-1][1].plot(x_line, y_line, '-r')
    if x_star is not None:
        plt.figs_axes[-1][1].plot(x_star, y_star, marker='*', markersize=16, color='r')
        if not __debug__:
            plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_mod_nan', comment, timestamp=False)

    plt.show()


@timeit
def dab_iterate_1D(mod='zvs', sim=False, save_all=False, zvs_plot_limit=1, iter='Cpar'):
    """
    Run the complete optimization procedure and save the results in a file
    """
    ## Set the basic DAB Specification
    dab = ds.dab_ds_default_Gv8_sim()
    # Assuming theoretical transformer
    # dab = ds.dab_ds_default_Gv8_sim_n4()
    # Assumption for tests
    if iter == 'Lc':
        dab.t_dead1 = 150e-9
        dab.t_dead2 = 150e-9
    if iter == 'Ls':
        dab.t_dead1 = 150e-9
        dab.t_dead2 = 150e-9
    if iter == 'tdead':
        # Assume some sane Lc values
        dab.Lc1 = 750e-6
        dab.Lc2 = 611e-6 / (dab.n ** 2)
        # dab.Lc2 = 55e-9
    if iter == 'Cpar':
        dab.Lc1 = 800e-6
        dab.t_dead1 = 150e-9
        dab.t_dead2 = 150e-9

    # For Sim
    if sim:
        dab.V2_step = 7
        dab.P_step = 7
    # Generate meshes
    dab.gen_meshes()

    if iter != 'Cpar':
        # Overestimate Coss and Cpar
        coss_1 = dab.coss_1 + dab.C_HB11
        coss_2 = dab.coss_2 + dab.C_HB22

    # Iteration params
    # zvs_plot_limit = 0.9
    steps = 50
    # testing...
    if sim:
        steps = 12  # ~200min
        # steps = 2  # for fast test

    if iter == 'Lc':
        # Lc1
        x_min = 400e-6
        x_max = 1400e-6
        # Lc2
        y_min = 400e-6 / (dab.n ** 2)
        y_max = 1400e-6 / (dab.n ** 2)
    if iter == 'Ls':
        # Ls
        x_min = 60e-6
        x_max = 120e-6
        # Lc1
        y_min = 400e-6
        y_max = 1400e-6
    if iter == 'tdead':
        # tdead1
        x_min = 50e-9
        x_max = 200e-9
        # tdead2
        y_min = 50e-9
        y_max = 200e-9
        steps = 7
    if iter == 'Cpar':
        steps = 50  # ~60min
        # Cpar
        x_min = 1e-12
        x_max = 100e-12
        y_min = 1e-12
        y_max = 100e-12

    # Set file names
    directory = '../dab_modulation_output/'

    # Set names in case we want to save something
    name_pre = 'mod_' + mod + '_'

    if iter == 'Lc':
        name = 'Iter_Lc1{}-{}uH_Lc2{}-{}uH'.format(int(x_min * 1e6), int(x_max * 1e6), int(y_min * 1e6),
                                                   int(y_max * 1e6))
    if iter == 'Ls':
        name = 'Iter_Ls{}-{}uH_Lc12_{}-{}uH'.format(int(x_min * 1e6), int(x_max * 1e6), int(y_min * 1e6),
                                                    int(y_max * 1e6))
    if iter == 'tdead':
        name = 'Iter_tdead1_{}-{}ns_tdead2_{}-{}ns'.format(int(x_min * 1e9), int(x_max * 1e9), int(y_min * 1e9),
                                                           int(y_max * 1e9))
    if iter == 'Cpar':
        name = 'Iter_Cpar{}-{}pF'.format(int(x_min * 1e12), int(x_max * 1e12))

    comment = 'Iteration of ' + name_pre + name
    fname = name_pre + name + '_ZVS-Coverage'

    # Create new dir for all files
    directory = directory + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name_pre + name
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    if not __debug__:
        if not os.path.exists(directory):
            os.mkdir(directory)

    ## Set sim defaults
    simfilepath = '../circuits/DAB_MOSFET_Modulation_v8.ipes'
    timestep = 1e-9
    simtime = 50e-6
    timestep_pre = 50e-9
    simtime_pre = 5e-3
    # geckoport = 43036
    # Automatically select a free port
    geckoport = 0
    geckoport = sim_gecko.get_free_port(43256, 43320)

    mod_keys = []

    # Create result meshes
    results = ds.DAB_Data()
    # Create meshgrid for iter values
    if iter == 'Lc':
        results.mesh_x, results.mesh_y = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
        results.mesh_x_name = 'Lc1'
        results.mesh_y_name = 'Lc2'
    if iter == 'Ls':
        results.mesh_x, results.mesh_y = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
        results.mesh_x_name = 'Ls'
        results.mesh_y_name = 'Lc1'
    if iter == 'tdead':
        results.mesh_x, results.mesh_y = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
        results.mesh_x_name = 't_dead1'
        results.mesh_y_name = 't_dead2'
    if iter == 'Cpar':
        results.mesh_x, results.mesh_y = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
        results.mesh_x = np.linspace(x_min, x_max, steps)

    sim_keys = ['p_sw1', 'p_cond1', 'p_sw2', 'p_cond2', 'v_ds_S11_sw_on', 'v_ds_S23_sw_on', 'i_HF1_S11_sw_on',
                'i_HF2_S23_sw_on']
    for key in sim_keys:
        results['iter_sim_' + key] = np.full_like(results.mesh_x, np.nan)
    # result_keys = ['iter_zvs_coverage_mod', 'iter_zvs_coverage_sim', 'iter_i_HF1_total_mean_sim', 'iter_', 'iter_', 'iter_', 'iter_', 'iter_', 'iter_', 'iter_']
    results.iter_mod_zvs_coverage = np.full_like(results.mesh_x, np.nan)
    results.iter_mod_contains_nan = np.full_like(results.mesh_x, np.nan)
    results.iter_sim_zvs_coverage = np.full_like(results.mesh_x, np.nan)
    results.iter_sim_i_HF1_total_mean = np.full_like(results.mesh_x, np.nan)

    if sim and not __debug__:
        # Logging
        log = db.log(filename=os.path.join(directory, 'dab_opt.log'))
        log.info('Starting L iteration: ' + name)
    else:
        info('Starting L iteration: ' + name)

    # Progressbar init
    # Calc total number of iterations to simulate
    it_total = results.mesh_x.size
    pbar = tqdm(total=it_total)

    # for Ls in np.linspace(Ls_min, Ls_max, steps):
    #     for Lc in np.linspace(Lc_min, Lc_max, steps):
    for vec in np.ndindex(results.mesh_x.shape):
        if iter == 'Lc':
            # Set Lc1 and Lc2 according to assumption
            Ls = dab.Ls
            Lc1 = results.mesh_x[vec]
            Lc2 = results.mesh_y[vec]
            t_dead1 = dab.t_dead1
            t_dead2 = dab.t_dead2
            # Set names in case we want to save something
            _name = 'Ls_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(Ls * 1e6),
                                                         int(Lc1 * 1e6),
                                                         int(Lc2 * 1e6))
        if iter == 'Ls':
            # Set Ls and Lc1/2 according to assumption
            Ls = results.mesh_x[vec]
            Lc1 = results.mesh_y[vec]
            Lc2 = results.mesh_y[vec] / (dab.n ** 2)
            t_dead1 = dab.t_dead1
            t_dead2 = dab.t_dead2
            # Set names in case we want to save something
            _name = 'Ls_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(Ls * 1e6),
                                                         int(Lc1 * 1e6),
                                                         int(Lc2 * 1e6))
        if iter == 'tdead':
            # Set tdead1/2 according to assumption
            Ls = dab.Ls
            Lc1 = dab.Lc1
            Lc2 = dab.Lc2
            t_dead1 = round(results.mesh_x[vec], 9)
            t_dead2 = round(results.mesh_y[vec], 9)
            # Set names in case we want to save something
            _name = 'tdead1_{}ns__tdead2_{}ns'.format(int(t_dead1 * 1e9),
                                                      int(t_dead2 * 1e9))
            # Important to have timestep always smaller than any time var in simulation model
            if (t_dead1 < timestep_pre) or (t_dead2 < timestep_pre):
                timestep_pre = t_dead1 if t_dead1 < t_dead2 else t_dead2
            info(timestep_pre, t_dead1, t_dead2)
        if iter == 'Cpar':
            Ls = dab.Ls
            Lc1 = dab.Lc1
            Lc2 = dab.Lc2
            t_dead1 = dab.t_dead1
            t_dead2 = dab.t_dead2
            C_HB1 = results.mesh_x[vec]
            _name = 'C_HB1_{}pF'.format(int(C_HB1 * 1e12))
            # Overestimate Coss and Cpar
            coss_1 = dab.coss_1 * 2 + C_HB1 * 2
            coss_2 = dab.coss_2 * 2 + C_HB1 * 2

        ## Modulation Calculation
        if mod == 'zvs':
            da_mod = mod_zvs.calc_modulation(dab.n,
                                             Ls,
                                             Lc1,
                                             Lc2,
                                             dab.fs,
                                             coss_1,
                                             coss_2,
                                             dab.mesh_V1,
                                             dab.mesh_V2,
                                             dab.mesh_P)
            # Save coverage for plot
            results.iter_mod_zvs_coverage[vec] = da_mod['zvs_coverage']
        if mod == 'sps':
            da_mod = mod_sps.calc_modulation(dab.n,
                                             Ls,
                                             dab.fs,
                                             dab.mesh_V1,
                                             dab.mesh_V2,
                                             dab.mesh_P)
            # Save coverage for plot
            results.iter_mod_zvs_coverage[vec] = np.nan
        # results.iter_mod_contains_nan[vec] = np.isnan(da_mod['tau1']).any()
        results.iter_mod_contains_nan[vec] = np.mean(np.isnan(da_mod['tau1']))

        if sim:
            # Simulation
            da_sim = sim_gecko.start_sim(dab.mesh_V1,
                                         dab.mesh_V2,
                                         dab.mesh_P,
                                         da_mod['phi'],
                                         da_mod['tau1'],
                                         da_mod['tau2'],
                                         t_dead1,
                                         t_dead2,
                                         dab.fs,
                                         Ls,
                                         Lc1,
                                         Lc2,
                                         dab.n,
                                         dab.temp,
                                         simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport=geckoport,
                                         C_HB1=C_HB1,
                                         C_HB2=C_HB1)

            # Save coverage for plot
            results.iter_sim_zvs_coverage[vec] = da_sim['zvs_coverage_notnan']
            results.iter_sim_i_HF1_total_mean[vec] = da_sim['i_HF1_total_mean']
            for key in sim_keys:
                results['iter_sim_' + key][vec] = np.nanmean(da_sim[key])

            if ((results.iter_sim_zvs_coverage[vec] >= zvs_plot_limit) or save_all) and not __debug__:
                log.info('\n' + _name)
                log.info('Calculation ZVS coverage: {}'.format(results.iter_mod_zvs_coverage[vec]))
                log.info('Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(da_sim['zvs_coverage'],
                                                                                            da_sim['zvs_coverage1'],
                                                                                            da_sim['zvs_coverage2']))
                log.info('Simulation I_1-total-mean: {}'.format(da_sim['i_HF1_total_mean']))
                # Unpack the results
                dab.append_result_dict(da_mod, name_pre='mod_zvs_' + _name + '_')
                dab.append_result_dict(da_sim, name_pre='sim_zvs_' + _name + '_')

            if (results.iter_sim_zvs_coverage[vec] >= zvs_plot_limit) and not __debug__:
                ## remember for Plotting
                mod_keys.append('zvs_' + _name)

        # Progressbar update, default increment +1
        pbar.update()

    # Progressbar end
    pbar.close()

    if not __debug__:
        # numpy saves everything for us in a handy zip file
        results.save_to_file(directory, name_pre + name, False, comment)
        results.pprint_to_file(os.path.join(directory, name_pre + name + 'pprint.txt'))
        # Readable DAB Specs except right Ls,Lc1,Lc2
        dab.save_to_file(directory=directory, name='dab_' + name, comment=name, timestamp=False)
        dab.pprint_to_file(os.path.join(directory, 'dab_' + name + '_pprint.txt'))

    if sim:
        ## Plotting
        # FIXME This may take a long time
        info('Start plotting dab for selected ZVS areas. This may take some time.')
        plot_mod_sim(dab, name, comment, directory, mod_keys, show_plot=False)

    # Find a common z_min
    z_min = round(float(min(np.nanmin(results.iter_mod_zvs_coverage), np.nanmin(results.iter_sim_zvs_coverage))),
                  ndigits=1)
    debug('z_min: {}'.format(z_min))
    debug('z_max mod, sim: {}, {}'.format(np.nanmax(results.iter_mod_zvs_coverage),
                                          np.nanmax(results.iter_sim_zvs_coverage)))

    if iter == 'Lc':
        x = results.mesh_x * 1e6
        y = results.mesh_y * 1e6
        xlabel = r'$L_\mathrm{1} \:/\: \mathrm{\mu H}$'
        ylabel = r'$L_\mathrm{2} \:/\: \mathrm{\mu H}$'
    if iter == 'Ls':
        x = results.mesh_x * 1e6
        y = results.mesh_y * 1e6
        xlabel = r'$L_\mathrm{s} \:/\: \mathrm{\mu H}$'
        ylabel = r'$L_\mathrm{1},\, L^\prime_\mathrm{2} \:/\: \mathrm{\mu H}$'
    if iter == 'tdead':
        x = results.mesh_x * 1e9
        y = results.mesh_y * 1e9
        xlabel = r'$t_\mathrm{dead1} \:/\: \mathrm{ns}$'
        ylabel = r'$t_\mathrm{dead2} \:/\: \mathrm{ns}$'
    if iter == 'Cpar':
        x = results.mesh_x * 1e12
        y = results.mesh_y * 1e12
        xlabel = r'$C_\mathrm{par} \:/\: \mathrm{pF}$'
        ylabel = r'$C_\mathrm{par} \:/\: \mathrm{pF}$'

    ## Plotting the zvs coverage
    plt = plot_dab.Plot_DAB(latex=False, figsize=(5, 5))
    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS Coverage')
    plt.subplot(x,
                results.iter_mod_zvs_coverage,
                ax=plt.figs_axes[-1][1],
                xlabel=xlabel,
                ylabel=r'ZVS Abdeckung')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_mod', comment, timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS Coverage Simulation')
    plt.subplot(x,
                results.iter_sim_zvs_coverage,
                ax=plt.figs_axes[-1][1],
                xlabel=xlabel,
                ylabel=r'ZVS Abdeckung')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_sim', comment, timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF1 total mean Simulation')
    plt.subplot(x,
                results.iter_sim_i_HF1_total_mean,
                ax=plt.figs_axes[-1][1],
                xlabel=xlabel,
                ylabel=r'$\overline{I}_\mathrm{1} \:/\: \mathrm{A}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF1_total_mean_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF1_S11_sw_on mean Simulation')
    plt.subplot(x,
                results.iter_sim_i_HF1_S11_sw_on,
                ax=plt.figs_axes[-1][1],
                xlabel=xlabel,
                ylabel=r'$\overline{i}_\mathrm{1,S11,sw-on} \:/\: \mathrm{A}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF1_S11_sw_on_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF2_S23_sw_on mean Simulation')
    plt.subplot(x,
                results.iter_sim_i_HF2_S23_sw_on,
                ax=plt.figs_axes[-1][1],
                xlabel=xlabel,
                ylabel=r'$\overline{i}_\mathrm{2,S23,sw-on} \:/\: \mathrm{A}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF2_S23_sw_on_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='sw loss')
    plt.subplot(x,
                results.iter_sim_p_sw1 + results.iter_sim_p_sw2,
                ax=plt.figs_axes[-1][1],
                xlabel=xlabel,
                ylabel=r'$\overline{P}_\mathrm{sw-loss} \:/\: \mathrm{W}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_p-sw_sim', comment, timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='cond loss Simulation')
    plt.subplot(x,
                results.iter_sim_p_cond1 + results.iter_sim_p_cond2,
                ax=plt.figs_axes[-1][1],
                xlabel=xlabel,
                ylabel=r'$\overline{P}_\mathrm{cond-loss} \:/\: \mathrm{W}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_p-cond_sim', comment, timestamp=False)

    if not (results.iter_mod_contains_nan == results.iter_mod_contains_nan[0]).all():
        plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS NAN')
        plt.subplot(x,
                    results.iter_mod_contains_nan,
                    ax=plt.figs_axes[-1][1],
                    xlabel=xlabel,
                    ylabel=r'NAN Abdeckung')
        # plt.figs_axes[-1][0].tight_layout()
        if not __debug__:
            plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_mod_nan', comment, timestamp=False)

    plt.show()


def post_iter_plot(dab_file, iter_file):
    # Loading
    dab_file = os.path.expanduser(dab_file)
    dab_file = os.path.expandvars(dab_file)
    dab_file = os.path.abspath(dab_file)
    dab = ds.load_from_file(dab_file)
    # dab.pprint()

    # Loading Iter
    iter_file = os.path.expanduser(iter_file)
    iter_file = os.path.expandvars(iter_file)
    iter_file = os.path.abspath(iter_file)
    results = ds.load_from_file(iter_file)

    # Set file/dir names
    directory = os.path.dirname(dab_file)
    # Create new dir for all files
    directory = os.path.join(directory, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + 'Replot')
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Retrieve some values
    if str(results.mesh_x_name) == 'Lc1':
        iter = 'Lc'
    if str(results.mesh_x_name) == 'Ls':
        iter = 'Ls'

    x_min = np.min(results.mesh_x)
    x_max = np.max(results.mesh_x)
    y_min = np.min(results.mesh_y)
    y_max = np.max(results.mesh_y)

    # Set names in case we want to save something
    name_pre = 'mod_zvs_'

    if iter == 'Lc':
        name = 'Iter_Lc1{}-{}uH_Lc2{}-{}uH'.format(int(x_min * 1e6), int(x_max * 1e6), int(y_min * 1e6),
                                                   int(y_max * 1e6))
    if iter == 'Ls':
        name = 'Iter_Ls{}-{}uH_Lc12_{}-{}uH'.format(int(x_min * 1e6), int(x_max * 1e6), int(y_min * 1e6),
                                                    int(y_max * 1e6))
    if iter == 'tdead':
        name = 'Iter_tdead1_{}-{}ns_tdead2_{}-{}ns'.format(int(x_min * 1e9), int(x_max * 1e9), int(y_min * 1e9),
                                                           int(y_max * 1e9))

    comment = 'Iteration of ' + name_pre + name
    fname = name_pre + name + '_ZVS-Coverage'

    ## Plotting
    info('Start plotting iterations...')

    # Find a common z_min
    z_min = round(float(min(np.nanmin(results.iter_mod_zvs_coverage), np.nanmin(results.iter_sim_zvs_coverage))),
                  ndigits=1)
    debug('z_min: {}'.format(z_min))
    debug('z_max mod, sim: {}, {}'.format(np.nanmax(results.iter_mod_zvs_coverage),
                                          np.nanmax(results.iter_sim_zvs_coverage)))

    if iter == 'Lc':
        x = results.mesh_x * 1e6
        y = results.mesh_y * 1e6
        xlabel = r'$L_\mathrm{1} \:/\: \mathrm{\mu H}$'
        ylabel = r'$L_\mathrm{2} \:/\: \mathrm{\mu H}$'
    if iter == 'Ls':
        x = results.mesh_x * 1e6
        y = results.mesh_y * 1e6
        xlabel = r'$L \:/\: \mathrm{\mu H}$'
        ylabel = r'$L_\mathrm{1},\, L^\prime_\mathrm{2} \:/\: \mathrm{\mu H}$'
    if iter == 'tdead':
        x = results.mesh_x * 1e9
        y = results.mesh_y * 1e9
        xlabel = r'$t_\mathrm{dead1} \:/\: \mathrm{ns}$'
        ylabel = r'$t_\mathrm{dead2} \:/\: \mathrm{ns}$'

    ## Plotting the zvs coverage
    plt = plot_dab.Plot_DAB(latex=False, figsize=(5, 5))
    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS Coverage')
    plt.subplot_contourf(x,
                         y,
                         results.iter_mod_zvs_coverage,
                         z_min=z_min,
                         z_max=1,
                         num_cont_lines=12,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'ZVS Abdeckung')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_mod', comment, timestamp=False)
    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS Coverage Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_zvs_coverage,
                         z_min=z_min,
                         z_max=1,
                         num_cont_lines=12,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'ZVS Abdeckung')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_sim', comment, timestamp=False)
    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF1 total mean Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_i_HF1_total_mean,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{I}_\mathrm{1} \:/\: \mathrm{A}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF1_total_mean_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF1_S11_sw_on mean Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_i_HF1_S11_sw_on,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{i}_\mathrm{1,S11,sw-on} \:/\: \mathrm{A}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF1_S11_sw_on_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='i_HF2_S23_sw_on mean Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_i_HF2_S23_sw_on,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{i}_\mathrm{2,S23,sw-on} \:/\: \mathrm{A}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_i_HF2_S23_sw_on_sim', comment,
                     timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='sw loss')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_p_sw1 + results.iter_sim_p_sw2,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{P}_\mathrm{sw-loss} \:/\: \mathrm{W}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_p-sw_sim', comment, timestamp=False)

    plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='cond loss Simulation')
    plt.subplot_contourf(x,
                         y,
                         results.iter_sim_p_cond1 + results.iter_sim_p_cond2,
                         ax=plt.figs_axes[-1][1],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         title=r'$\overline{P}_\mathrm{cond-loss} \:/\: \mathrm{W}$')
    # plt.figs_axes[-1][0].tight_layout()
    if not __debug__:
        plt.save_fig(plt.figs_axes[-1][0], directory, name_pre + name + '_p-cond_sim', comment, timestamp=False)

    if not (results.iter_mod_contains_nan == results.iter_mod_contains_nan[0]).all():
        plt.new_fig(nrows=1, ncols=1, sharex=False, sharey=False, tab_title='ZVS NAN')
        plt.subplot_contourf(x,
                             y,
                             results.iter_mod_contains_nan,
                             # results.iter_mod_zvs_coverage,
                             # nan_matrix=nanmatrix,
                             ax=plt.figs_axes[-1][1],
                             xlabel=xlabel,
                             ylabel=ylabel,
                             title=r'NAN Abdeckung')
        # plt.figs_axes[-1][0].tight_layout()
        if not __debug__:
            plt.save_fig(plt.figs_axes[-1][0], directory, fname + '_mod_nan', comment, timestamp=False)

    plt.show()


def post_simresults_plot(dab_file, mod_keys=('sps', 'mcl', 'zvs')):
    # Loading
    dab_file = os.path.expanduser(dab_file)
    dab_file = os.path.expandvars(dab_file)
    dab_file = os.path.abspath(dab_file)
    dab = ds.load_from_file(dab_file)
    # dab.pprint()

    # Add some missing values
    for mod in mod_keys:
        dab['sim_' + mod + '_I1_squared_total_mean'] = np.nanmean(dab['sim_' + mod + '_i_HF1'] ** 2)

    # Set file/dir names
    directory = os.path.dirname(dab_file)
    file = os.path.basename(dab_file)

    # Set name for figs
    # dab_sim_Gv8_original-setup_Ls_85uH__Lc1_25620uH__Lc2_68uH_v1-v50-p38.npz
    # debug(file.split('_', 1)[1].split('_v', 1)[0])
    name = file.split('_', 1)[1].split('_v', 1)[0]

    # Create new dir for all files
    directory = os.path.join(directory, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + 'Replot')
    if not os.path.exists(directory):
        os.mkdir(directory)

    comment = str(dab._comment)
    plot_mod_sim(dab, name, comment, directory, mod_keys, False, 'dab_replot.log')
    # plot_mod_sim(dab, name, comment, directory, mod_keys, True, 'dab_replot.log')
    # plot_mod_sim(dab, name, comment, directory, ['zvs'], False, 'dab_replot.log')


@timeit
def trial_plot_animation():
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


def _main_dummy():
    return


# ---------- MAIN ----------
if __name__ == '__main__':
    info("Start of DAB Optimizer ...")
    # Do some basic init like logging, args, etc.
    main_init()

    # Only modulation calculation
    dab_mod_save()

    # Generate simulation data
    # dab_sim_save()
    # dab_sim_save_zvs()
