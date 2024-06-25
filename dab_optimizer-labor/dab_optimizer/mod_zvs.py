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

"""
## ATTENTION - Naming Conflict
Modulation names "mode 1", "mode 2" and "mode 5" are not the same in different papers!
Names used here are:
High Power Flow: mode 1+ : -tau1 + pi <= phi <= tau2
High Power Flow: mode 1- : -tau1 <= phi <= tau2 - pi
Low  Power Flow: mode 2  : tau2 - tau1 <= phi <= 0

## Definitions
All small letters are time dependent values.
A trailing _ indicates a value transformed to the primary side, e.g. v2_ or iHF2_
All secondary side values are transformed to the primary side, so V2 becomes V2_ and so forth.
Formula numbers refer to [2].
Primary side values are named with 1 and secondary side with 2, e.g. I1 is primary side DC input current.

d: the primary side referred voltage conversion ratio: d = Vdc2 / Vdc1 = V2 / V1
V2_ = n * V2
n: Transformer winding ratio n = n1 / n2
ws: omega_s = 2 pi fs
Q_AB_req1: Q_AB_req_p but with changed naming according to side 1 and 2 naming scheme.
"""

import numpy as np
import math
import dab_opt
import dab_datasets as ds
from debug_tools import *

# The dict keys this modulation will return
MOD_KEYS = ['mod_zvs_phi', 'mod_zvs_tau1', 'mod_zvs_tau2', 'mod_zvs_mask_zvs', 'mod_zvs_mask_m1p', 'mod_zvs_mask_m1n',
            'mod_zvs_mask_m2']


@timeit
def calc_modulation(n, Ls, Lc1, Lc2, fs: np.ndarray | int | float, Coss1: np.ndarray, Coss2: np.ndarray,
                    V1: np.ndarray, V2: np.ndarray, P: np.ndarray) -> dict:
    """
    OptZVS (Optimal ZVS) Modulation calculation, which will return phi, tau1 and tau2

    :param n: Transformer turns ratio n1/n2.
    :param Ls: DAB converter series inductance. (Must not be zero!)
    :param Lc1: Side 1 commutation inductance. Use np.inf it not present.
    :param Lc2: Side 2 commutation inductance. Use np.inf it not present. (Must not be zero!)
    :param fs: Switching frequency, can be a fixed value or a meshgrid with same shape as the other meshes.
    :param Coss1: Side 1 MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
    :param Coss2: Side 2 MOSFET Coss(Vds) curve from Vds=0V to >= V2_max. Just one row with Coss data and index = Vds.
    :param V1: Input voltage meshgrid (voltage on side 1).
    :param V2: Output voltage meshgrid (voltage on side 2).
    :param P: DAB input power meshgrid (P=V1*I1).
    :return: dict with phi, tau1, tau2, masks (phi has First-Falling-Edge alignment!)
    """

    # TODO DUMMY
    phi = np.full_like(V1, np.pi)
    tau1 = np.full_like(V1, np.pi)
    tau2 = np.full_like(V1, np.pi)
    zvs = np.greater_equal(P, 1000)
    _m1p_mask = np.full_like(V1, np.nan)
    _m1n_mask = np.full_like(V1, np.nan)
    _m2_mask = np.full_like(V1, np.nan)

    Lc2_ = Lc2 * n ** 2
    ws = 2 * np.pi * fs
    # TODO make Q_AB like a V mesh
    Q_AB_req1 = _integrate_Coss(Coss1) * 1.05
    Q_AB_req2 = _integrate_Coss(Coss2) * 1.05
    # FIXME FAKE DATA
    Q_AB_req1 = np.full_like(V1, Q_AB_req1[700])
    Q_AB_req2 = np.full_like(V1, Q_AB_req2[235])
    V2_ = V2 * n
    I1 = P / V1
    phi, tau1, tau2 = _calc_interval_I(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)




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


def _integrate_Coss(coss):
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
    print(coss, coss_int)
    # get an qoss vector that has the resolution 1V from 0 to V_max
    v_vec = np.arange(coss_int.shape[0])
    # get an qoss vector that fits the mesh_V scale
    # v_vec = np.linspace(V_min, V_max, int(V_step))
    qoss = coss_int(v_vec)
    # Scale from pC to nC
    qoss = qoss / 1000

    # TODO make qoss like a V mesh

    # np.savetxt('qoss.csv', qoss, delimiter=';')
    return qoss


@timeit
def _calc_interval_I(n, Ls, Lc1, Lc2_, ws: np.ndarray | int | float, Q_AB_req1: np.ndarray, Q_AB_req2: np.ndarray,
                    V1: np.ndarray, V2_: np.ndarray, I1: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 2 Modulation calculation, which will return phi, tau1 and tau2
    """
    ## Predefined Terms
    e1 = V2_ * Q_AB_req2 * ws

    e2 = n * V1 * np.pi * I1

    e3 = n * (V2_ * (Lc2_ + Ls) - V1 * Lc2_)

    e4 = 2 * n * np.sqrt(Q_AB_req1 * Ls * np.power(ws, 2) * V1 * Lc1 * (Lc1 + Ls))

    e5 = Ls * Lc2_ * ws * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    ## Solution for interval I (mode 2)
    tau1 = (np.sqrt(2) * (Lc1 * np.sqrt(V2_ * e3 * e5) + e4 * e3 * 1 / n)) / (V1 * e3 * (Lc1 + Ls))

    tau2 = np.sqrt((2 * e5) / (V2_ * e3))

    phi = (tau2 - tau1) / 2 + (I1 * ws * Ls * np.pi) / (tau2 * V2_)

    debug(phi, tau1, tau2)
    return phi, tau1, tau2


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
    dab_specs.Lc1 = np.inf
    dab_specs.Lc2 = np.inf
    dab_specs.fs_nom = 200000

    # Object to store all generated data
    dab_results = ds.DAB_Results()
    # gen meshes
    dab_results.gen_meshes(
        dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step,
        dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step,
        dab_specs.P_min, dab_specs.P_max, dab_specs.P_step)

    # Import Coss curves
    csv_file = 'C:/Users/vijay/Desktop/UPB/Thesis/Coss_C3M0120100J.csv'

    dab_specs['coss_C3M0120100J'] = dab_opt.import_Coss(csv_file)
    print(dab_specs['coss_C3M0120100J'])
    # Generate Qoss matrix
    # dab_specs['qoss_C3M0120100J'] = dab_opt.integrate_Coss(dab_specs['coss_C3M0120100J'])

    # Modulation Calculation
    da_mod = calc_modulation(dab_specs.n,
                             dab_specs.L_s,
                             dab_specs.Lc1,
                             dab_specs.Lc2,
                             dab_specs.fs_nom,
                             dab_specs['coss_C3M0120100J'],
                             dab_specs['coss_C3M0120100J'],
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
