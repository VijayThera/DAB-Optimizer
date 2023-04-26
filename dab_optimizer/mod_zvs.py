#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

"""
Calculation of the Modulation for a DAB (Dual Active Bridge).

This module calculates the **OptZVS (Optimal ZVS) Modulation** according to the Paper [IEEE][1] and PhD Thesis [2].
It enables ZVS wherever possible and minimizes the inductor rms current I_L.
A Matrix *mask_zvs* will be returned indicating where ZVS is possible or not.

It was tried to be as close as possible to the depicted algorithm but a around formula (25) and (23) had been made.

[1]: https://ieeexplore.ieee.org/document/7762886 (
J. Everts, "Closed-Form Solution for Efficient ZVS Modulation of DAB Converters,"
in IEEE Transactions on Power Electronics, vol. 32, no. 10, pp. 7561-7576, Oct. 2017, doi: 10.1109/TPEL.2016.2633507.
)
[2]: https://kuleuven.limo.libis.be/discovery/fulldisplay?docid=lirias1731206&context=SearchWebhook&vid=32KUL_KUL:Lirias&search_scope=lirias_profile&tab=LIRIAS&adaptor=SearchWebhook&lang=en (
Everts, Jordi / Driesen, Johan
Modeling and Optimization of Bidirectional Dual Active Bridge AC-DC Converter Topologies
(Modellering en optimalisatie van bidirectionele dual active bridge AC-DC convertor topologieën)
2014-04-11
)
"""

import numpy as np

import dab_datasets as ds
from debug_tools import *

# The dict keys this modulation will return
MOD_KEYS = ['mod_zvs_phi', 'mod_zvs_tau1', 'mod_zvs_tau2', 'mod_zvs_mask_zvs', 'mod_zvs_mask_m1p', 'mod_zvs_mask_m1n',
            'mod_zvs_mask_m2']


@timeit
def calc_modulation(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P) -> dict:
    """
    OptZVS (Optimal ZVS) Modulation calculation, which will return phi, tau1 and tau2

    :param n: Transformer turns ratio
    :param L_s: DAB converter inductance.
    :param fs_nom: Switching frequency
    :param mesh_V1: input voltage (voltage on side 1)
    :param mesh_V2: output voltage (voltage on side 2)
    :param mesh_P: DAB power (assuming a lossless DAB)
    :return: dict with phi, tau1, tau2, masks
    """

    # TODO DUMMY
    phi = np.full_like(mesh_V1, np.pi)
    tau1 = np.full_like(mesh_V1, np.pi)
    tau2 = np.full_like(mesh_V1, np.pi)
    zvs = np.greater_equal(mesh_P, 1000)
    _m1p_mask = np.full_like(mesh_V1, np.nan)
    _m1n_mask = np.full_like(mesh_V1, np.nan)
    _m2_mask = np.full_like(mesh_V1, np.nan)

    # Init return dict
    da_mod_results = dict()
    # Save the results in the dict
    # da_mod_results[MOD_KEYS[0]] = phi
    # Convert phi because the math from the paper uses Middle-Pulse alignment but we use First-Falling-Edge alignment!
    da_mod_results[MOD_KEYS[0]] = phi
    da_mod_results[MOD_KEYS[1]] = tau1
    da_mod_results[MOD_KEYS[2]] = tau2
    da_mod_results[MOD_KEYS[3]] = zvs
    da_mod_results[MOD_KEYS[4]] = _m1p_mask
    da_mod_results[MOD_KEYS[5]] = _m1n_mask
    da_mod_results[MOD_KEYS[6]] = _m2_mask

    return da_mod_results


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module CPM ...")

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

    info("\nStart Plotting\n")
    import plot_dab

    v1_middle = int(np.shape(dab_results.mesh_P)[1] / 2)

    plt = plot_dab.Plot_DAB()
    # Plot all modulation angles
    plt.new_fig(nrows=1, ncols=3, tab_title='DAB Modulation Angles')
    plt.plot_modulation(plt.figs_axes[-1],
                             dab_results.mesh_P[:, v1_middle, :],
                             dab_results.mesh_V2[:, v1_middle, :],
                             dab_results.mod_zvs_phi[:, v1_middle, :],
                             dab_results.mod_zvs_tau1[:, v1_middle, :],
                             dab_results.mod_zvs_tau2[:, v1_middle, :],
                             mask1=dab_results.mod_zvs_mask_tcm[:, v1_middle, :],
                             mask2=dab_results.mod_zvs_mask_cpm[:, v1_middle, :])

    # Plot animation for every V1 cross-section
    # for v1 in range(0, np.shape(dab_results.mesh_P)[1] - 1):
    #     print(v1)
    #     plt.plot_modulation(plt.figs_axes[-1],
    #                              dab_results.mesh_P[:, v1, :],
    #                              dab_results.mesh_V2[:, v1, :],
    #                              dab_results.mod_zvs_phi[:, v1, :],
    #                              dab_results.mod_zvs_tau1[:, v1, :],
    #                              dab_results.mod_zvs_tau2[:, v1, :],
    #                              mask1=dab_results.mod_zvs_mask_tcm[:, v1, :],
    #                              mask2=dab_results.mod_zvs_mask_cpm[:, v1, :])

    plt.show()
