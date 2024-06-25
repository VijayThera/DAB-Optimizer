#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

"""
Calculation of the Modulation for a DAB (Dual Active Bridge).

This module calculates the **CPS (Conventional Phase Shift) Modulation**
which is often also referred to as **SPS (Single-phase-shift) Modulation**.
"""

import numpy as np

import dab_datasets as ds
from debug_tools import *

# The dict keys this modulation will return
MOD_KEYS = ['mod_sps_phi', 'mod_sps_tau1', 'mod_sps_tau2']


@timeit
def calc_modulation(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P) -> dict:
    # Init return dict
    da_mod_results = dict()
    # Init tau with pi because they are constant for CPM
    da_mod_results[MOD_KEYS[1]] = np.full_like(mesh_V1, np.pi)
    da_mod_results[MOD_KEYS[2]] = np.full_like(mesh_V1, np.pi)

    # Calculate phase shift difference phi from input to output bridge
    # TODO maybe have to consider the case sqrt(<0). When does this happen?
    # maybe like this: mln_phi[mln_phi < 0] = np.nan
    da_mod_results[MOD_KEYS[0]] = np.pi / 2 * (1 - np.sqrt(1 - (8 * fs_nom * L_s * mesh_P) / (n * mesh_V1 * mesh_V2)))

    return da_mod_results


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module CPM ...")

    dab_specs = ds.DAB_Specification()
    dab_specs.V1_nom = 700
    dab_specs.V1_min = 600
    dab_specs.V1_max = 800
    dab_specs.V1_step = 3
    dab_specs.V2_nom = 235
    dab_specs.V2_min = 175
    dab_specs.V2_max = 295
    dab_specs.V2_step = 4
    dab_specs.P_min = 400
    dab_specs.P_max = 2200
    dab_specs.P_nom = 2000
    dab_specs.P_step = 5
    dab_specs.n = 2.99
    dab_specs.L_s = 84e-6
    dab_specs.L_m = 599e-6
    dab_specs.fs_nom = 200000

    # Object to store all generated data
    dab_results = ds.DAB_Results()
    # gen meshes
    dab_results.gen_meshes(
        dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step,
        dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step,
        dab_specs.P_min, dab_specs.P_max, dab_specs.P_step)

    # Modulation Calculation
    da_mod = calc_modulation(dab_specs.n,
                             dab_specs.L_s,
                             dab_specs.fs_nom,
                             dab_results.mesh_V1,
                             dab_results.mesh_V2,
                             dab_results.mesh_P)

    # Unpack the results
    dab_results.append_result_dict(da_mod)

    print("mod_sps_phi:", dab_results.mod_sps_phi, sep='\n')
    print("mod_sps_tau1:", dab_results.mod_sps_tau1, sep='\n')
    print("mod_sps_tau2:", dab_results.mod_sps_tau2, sep='\n')
    print("mod_sps_phi[0,0,0]", type(dab_results.mod_sps_phi[0, 0, 0]))
    print("mod_sps_tau1[0,0,0]", type(dab_results.mod_sps_tau1[0, 0, 0]))
    print("mod_sps_tau2[0,0,0]", type(dab_results.mod_sps_tau2[0, 0, 0]))

    info("\nStart Plotting\n")
    import plot_dab

    plt = plot_dab.Plot_DAB()
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='DAB Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, 1, :],
                             dab_results.mesh_V2[:, 1, :],
                             dab_results.mod_sps_phi[:, 1, :],
                             dab_results.mod_sps_tau1[:, 1, :],
                             dab_results.mod_sps_tau2[:, 1, :])
    plt.show()
