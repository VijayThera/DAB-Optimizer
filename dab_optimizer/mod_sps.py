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

    Dab_Specs = ds.DAB_Specification()
    Dab_Specs.V1_nom = 700
    Dab_Specs.V1_min = 600
    Dab_Specs.V1_max = 800
    Dab_Specs.V1_step = 3
    Dab_Specs.V2_nom = 235
    Dab_Specs.V2_min = 175
    Dab_Specs.V2_max = 295
    Dab_Specs.V2_step = 4
    Dab_Specs.P_min = 400
    Dab_Specs.P_max = 2200
    Dab_Specs.P_nom = 2000
    Dab_Specs.P_step = 5
    Dab_Specs.n = 2.99
    Dab_Specs.L_s = 84e-6
    Dab_Specs.L_m = 599e-6
    Dab_Specs.fs_nom = 200000

    # Object to store all generated data
    Dab_Results = ds.DAB_Results()
    # gen meshes
    Dab_Results.gen_meshes(
        Dab_Specs.V1_min, Dab_Specs.V1_max, Dab_Specs.V1_step,
        Dab_Specs.V2_min, Dab_Specs.V2_max, Dab_Specs.V2_step,
        Dab_Specs.P_min, Dab_Specs.P_max, Dab_Specs.P_step)

    # Modulation Calculation
    da_mod = calc_modulation(Dab_Specs.n,
                             Dab_Specs.L_s,
                             Dab_Specs.fs_nom,
                             Dab_Results.mesh_V1,
                             Dab_Results.mesh_V2,
                             Dab_Results.mesh_P)

    # Unpack the results
    Dab_Results.append_result_dict(da_mod)

    print("mod_sps_phi:", Dab_Results.mod_sps_phi, sep='\n')
    print("mod_sps_tau1:", Dab_Results.mod_sps_tau1, sep='\n')
    print("mod_sps_tau2:", Dab_Results.mod_sps_tau2, sep='\n')
    print("mod_sps_phi[0,0,0]", type(Dab_Results.mod_sps_phi[0, 0, 0]))
    print("mod_sps_tau1[0,0,0]", type(Dab_Results.mod_sps_tau1[0, 0, 0]))
    print("mod_sps_tau2[0,0,0]", type(Dab_Results.mod_sps_tau2[0, 0, 0]))

    info("\nStart Plotting\n")
    import plot_dab

    Plot_Dab = plot_dab.Plot_DAB()
    # Plot all modulation angles
    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='DAB Modulation Angles')
    Plot_Dab.plot_modulation(Plot_Dab.figs_axes[-1],
                             Dab_Results.mesh_P[:, 1, :],
                             Dab_Results.mesh_V2[:, 1, :],
                             Dab_Results.mod_sps_phi[:, 1, :],
                             Dab_Results.mod_sps_tau1[:, 1, :],
                             Dab_Results.mod_sps_tau2[:, 1, :])
    Plot_Dab.show()
