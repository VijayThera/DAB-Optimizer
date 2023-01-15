#!/usr/bin/python3
# coding: utf-8
# python >= 3.10

import numpy as np

import classes_datasets as ds
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
    dab_results.mesh_V1, dab_results.mesh_V2, dab_results.mesh_P = np.meshgrid(
        np.linspace(dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step),
        np.linspace(dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step),
        np.linspace(dab_specs.P_min, dab_specs.P_max, dab_specs.P_step), sparse=False)

    # Modulation Calculation
    dab_results.mvvp_phi, dab_results.mvvp_tau1, dab_results.mvvp_tau2 = calc_modulation(dab_specs.n,
                                                                                         dab_specs.L_s,
                                                                                         dab_specs.fs_nom,
                                                                                         dab_results.mesh_V1,
                                                                                         dab_results.mesh_V2,
                                                                                         dab_results.mesh_P)

    print(dab_results.mvvp_phi)
    print("mvvp_phi[0,0,0]", type(dab_results.mvvp_phi[0, 0, 0]))
    print("mvvp_tau1[0,0,0]", type(dab_results.mvvp_tau1[0, 0, 0]))
    