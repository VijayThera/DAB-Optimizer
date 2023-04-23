#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

"""
Calculation of the Modulation for a DAB (Dual Active Bridge).

This file lists all formulas for the **OptZVS (Optimal ZVS) Modulation** according to the Paper [IEEE][1] and PhD Thesis [2].

All names are converted in such way that they can be python variables.

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
MOD_KEYS = ['mod_optzvs_phi', 'mod_optzvs_tau1', 'mod_optzvs_tau2', 'mod_optzvs_mask_zvs', 'mod_optzvs_mask_m1n', 'mod_optzvs_mask_m1p', 'mod_optzvs_mask_m2']


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

    Dab_Specs = ds.DAB_Specification()
    Dab_Specs.V1_nom = 700
    Dab_Specs.V1_min = 600
    Dab_Specs.V1_max = 800
    Dab_Specs.V1_step = 21 * 3
    Dab_Specs.V2_nom = 235
    Dab_Specs.V2_min = 175
    Dab_Specs.V2_max = 295
    Dab_Specs.V2_step = 25 * 3
    Dab_Specs.P_min = 400
    Dab_Specs.P_max = 2200
    Dab_Specs.P_nom = 2000
    Dab_Specs.P_step = 19 * 3
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

    info("\nStart Plotting\n")
    import plot_dab

    v1_middle = int(np.shape(Dab_Results.mesh_P)[1] / 2)

    Plot_Dab = plot_dab.Plot_DAB()
    # Plot all modulation angles
    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='DAB Modulation Angles')
    Plot_Dab.plot_modulation(Plot_Dab.figs_axes[-1],
                             Dab_Results.mesh_P[:, v1_middle, :],
                             Dab_Results.mesh_V2[:, v1_middle, :],
                             Dab_Results.mod_optzvs_phi[:, v1_middle, :],
                             Dab_Results.mod_optzvs_tau1[:, v1_middle, :],
                             Dab_Results.mod_optzvs_tau2[:, v1_middle, :],
                             mask1=Dab_Results.mod_optzvs_mask_tcm[:, v1_middle, :],
                             mask2=Dab_Results.mod_optzvs_mask_cpm[:, v1_middle, :])

    # Plot animation for every V1 cross-section
    # for v1 in range(0, np.shape(Dab_Results.mesh_P)[1] - 1):
    #     print(v1)
    #     Plot_Dab.plot_modulation(Plot_Dab.figs_axes[-1],
    #                              Dab_Results.mesh_P[:, v1, :],
    #                              Dab_Results.mesh_V2[:, v1, :],
    #                              Dab_Results.mod_optzvs_phi[:, v1, :],
    #                              Dab_Results.mod_optzvs_tau1[:, v1, :],
    #                              Dab_Results.mod_optzvs_tau2[:, v1, :],
    #                              mask1=Dab_Results.mod_optzvs_mask_tcm[:, v1, :],
    #                              mask2=Dab_Results.mod_optzvs_mask_cpm[:, v1, :])

    Plot_Dab.show()
