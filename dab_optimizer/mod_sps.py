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


@timeit
def calc_modulation(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P):
    # init 3d arrays
    # mvvp_phi = np.zeros_like(mesh_V1)
    # init these with pi because they are constant for CPM
    mvvp_tau1 = np.full_like(mesh_V1, np.pi)
    mvvp_tau2 = np.full_like(mesh_V1, np.pi)

    # Calculate phase shift difference from input to output bridge
    # TODO maybe have to consider the case sqrt(<0). When does this happen?
    # maybe like this: mln_phi[mln_phi < 0] = np.nan
    mvvp_phi = np.pi / 2 * (1 - np.sqrt(1 - (8 * fs_nom * L_s * mesh_P) / (n * mesh_V1 * mesh_V2)))

    return mvvp_phi, mvvp_tau1, mvvp_tau2


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
    Dab_Results.mod_phi, Dab_Results.mod_tau1, Dab_Results.mod_tau2 = calc_modulation(Dab_Specs.n,
                                                                                      Dab_Specs.L_s,
                                                                                      Dab_Specs.fs_nom,
                                                                                      Dab_Results.mesh_V1,
                                                                                      Dab_Results.mesh_V2,
                                                                                      Dab_Results.mesh_P)

    print("mod_phi:", Dab_Results.mod_phi, sep='\n')
    print("mod_tau1:", Dab_Results.mod_tau1, sep='\n')
    print("mod_tau2:", Dab_Results.mod_tau2, sep='\n')
    print("mod_phi[0,0,0]", type(Dab_Results.mod_phi[0, 0, 0]))
    print("mod_tau1[0,0,0]", type(Dab_Results.mod_tau1[0, 0, 0]))
    print("mod_tau2[0,0,0]", type(Dab_Results.mod_tau2[0, 0, 0]))
    