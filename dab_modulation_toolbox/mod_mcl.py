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

"""
Calculation of the Modulation for a DAB (Dual Active Bridge).

This module calculates the **MCL (Minimum Conduction Loss) Modulation** according to the Paper [IEEE][1].
It minimizes the inductor rms current I_L.

It was tried to be as close as possible to the depicted algorithm but a around formula (25) and (23) had been made.

[1]: https://ieeexplore.ieee.org/document/5776689 (
F. Krismer and J. W. Kolar, "Closed Form Solution for Minimum Conduction Loss Modulation of DAB Converters,"
in IEEE Transactions on Power Electronics, vol. 27, no. 1, pp. 174-188, Jan. 2012, doi: 10.1109/TPEL.2011.2157976.
)
"""

import numpy as np

import dab_datasets as ds
from debug_tools import *

# The dict keys this modulation will return
# MOD_KEYS = ['mod_mcl_phi', 'mod_mcl_tau1', 'mod_mcl_tau2', 'mod_mcl_mask_tcm', 'mod_mcl_mask_otm', 'mod_mcl_mask_cpm']
MOD_KEYS = ['phi', 'tau1', 'tau2', 'mask_tcm', 'mask_otm', 'mask_cpm']


@timeit
def calc_modulation(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P) -> dict:
    """
    MCL (Minimum Conduction Loss) Modulation calculation, which will return phi, tau1 and tau2

    :param n: Transformer turns ratio
    :param L_s: DAB converter inductance.
    :param fs_nom: Switching frequency
    :param mesh_V1: input voltage (voltage on side 1)
    :param mesh_V2: output voltage (voltage on side 2)
    :param mesh_P: DAB power (assuming a lossless DAB)
    :return: dict with phi, tau1, tau2, masks
    """

    # Reference voltage, any arbitrary voltage!
    V_ref = 100

    # Calculate normalized voltage and power values V1n, V2n, Pn
    V1n, V2n, Pn = _normalize_input_arrays(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P, V_ref)

    # Calculate Pn_max with (14)
    # TODO maybe check if inductor L_s is too big for the power P_max
    Pn_max = _Pn_max(V1n, V2n)

    # Limit Pn to +/- Pn_max
    Pn_no_lim = Pn  # saved just in case...
    Pn = _limit_Pn(Pn, Pn_max)

    # Determine Van and Vbn with (20)
    Van = np.full_like(V1n, np.nan)
    Vbn = np.full_like(V1n, np.nan)
    # input value mapping
    _transformation_mask = np.less_equal(V1n, V2n)
    # debug(_transformation_mask)
    # if np.less_equal(V1n, V2n):
    Van[_transformation_mask] = V1n[_transformation_mask]
    Vbn[_transformation_mask] = V2n[_transformation_mask]
    # else: if np.greater(V1n, V2n):
    Van[np.bitwise_not(_transformation_mask)] = V2n[np.bitwise_not(_transformation_mask)]
    Vbn[np.bitwise_not(_transformation_mask)] = V1n[np.bitwise_not(_transformation_mask)]

    # Calculate Pn_tcm,max with (22). Maximum power for TCM!
    Pn_tcmmax = np.pi / 2 * (np.power(Van, 2) * (Vbn - Van)) / Vbn

    # ***** Change in contrast to paper *****
    # Instead of fist checking the power limits for each modulation and only calculate each mod. partly,
    # all the modulations are calculated first even for useless areas, and we decide later which part is useful.
    # This should be faster and easier.

    # TCM: calculate phi, Da and Db with (21)
    phi_tcm, da_tcm, db_tcm = _calc_TCM(Van, Vbn, Pn)

    # OTM: calculate phi, Da and Db with (23) and (24)
    phi_otm, da_otm, db_otm = _calc_OTM(Van, Vbn, Pn)

    # CPM/SPS: calculate phi, Da and Db with (26)
    phi_cpm, da_cpm, db_cpm = _calc_CPM(Van, Vbn, Pn)

    # if abs(Pn) <= Pn_tcmmax: use TCM
    # _tcm_mask = np.full_like(Pn, np.nan)
    _tcm_mask = np.less_equal(np.abs(Pn), Pn_tcmmax)
    # debug('TCM MASK\n', _tcm_mask)

    # Calculate Pn_optmax with (25)
    # OTM for Pn_tcmmax < abs(Pn) <= Pn_optmax
    _otm_mask = np.less_equal(db_otm, 1 / 2)
    # CPM where db_otm > 1/2 or nan, this is simply the inverse
    _cpm_mask = np.bitwise_not(_otm_mask)
    # From the OTM mask we now "subtract" the TCM mask, that way where TCM is possible OTM mask is false too
    _otm_mask[_tcm_mask] = False
    # debug('OPT MASK\n', _otm_mask)
    # debug('CPM MASK\n', _cpm_mask)

    # Finally select the results according to their boundaries
    phi = np.full_like(Pn, np.nan)
    Da = np.full_like(Pn, np.nan)
    Db = np.full_like(Pn, np.nan)

    # use OTM: if Pn_tcmmax < abs(Pn) <= Pn_optmax
    phi[_otm_mask] = phi_otm[_otm_mask]
    Da[_otm_mask] = da_otm[_otm_mask]
    Db[_otm_mask] = db_otm[_otm_mask]

    # use CPM: if abs(Pn) > Pn_optmax
    phi[_cpm_mask] = phi_cpm[_cpm_mask]
    Da[_cpm_mask] = da_cpm[_cpm_mask]
    Db[_cpm_mask] = db_cpm[_cpm_mask]

    # use TCM: if abs(Pn) <= Pn_tcmmax
    phi[_tcm_mask] = phi_tcm[_tcm_mask]
    Da[_tcm_mask] = da_tcm[_tcm_mask]
    Db[_tcm_mask] = db_tcm[_tcm_mask]

    # Determine D1 and D2 with (20)
    D1 = np.full_like(Da, np.nan)
    D2 = np.full_like(Da, np.nan)
    # output value mapping
    # if np.less_equal(V1n, V2n):
    D1[_transformation_mask] = Da[_transformation_mask]
    D2[_transformation_mask] = Db[_transformation_mask]
    # else: if np.greater(V1n, V2n):
    D2[np.bitwise_not(_transformation_mask)] = Da[np.bitwise_not(_transformation_mask)]
    D1[np.bitwise_not(_transformation_mask)] = Db[np.bitwise_not(_transformation_mask)]

    # convert duty cycle D into radiant angle tau
    # FIXME: The duty cycle D=1/2 represents actually pi not pi/2! Therefore all Simulations before April 2023 are wrong!
    # FIXED that hard to find error now to D*2*pi:
    tau1 = D1 * 2 * np.pi
    tau2 = D2 * 2 * np.pi

    # Init return dict
    da_mod_results = dict()
    # Save the results in the dict
    # da_mod_results[MOD_KEYS[0]] = phi
    # Convert phi because the math from the paper uses Middle-Pulse alignment but we use First-Falling-Edge alignment!
    da_mod_results[MOD_KEYS[0]] = convert_phiM_to_phiF(phi, tau1, tau2)
    da_mod_results[MOD_KEYS[1]] = tau1
    da_mod_results[MOD_KEYS[2]] = tau2
    da_mod_results[MOD_KEYS[3]] = _tcm_mask
    da_mod_results[MOD_KEYS[4]] = _otm_mask
    da_mod_results[MOD_KEYS[5]] = _cpm_mask

    return da_mod_results


def _normalize_input_arrays(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P, V_ref: float = 100) -> [np.ndarray, np.ndarray,
                                                                                              np.ndarray]:
    """
    Normalize the given meshes to the reference voltage.
    The lower case "n" denotes that these values are normalized.

    :param n: Transformer turns ratio
    :param L_s: DAB converter inductance
    :param fs_nom: Switching frequency
    :param mesh_V1: input voltage (voltage on side 1)
    :param mesh_V2: output voltage (voltage on side 2)
    :param mesh_P: DAB power (assuming a lossless DAB)
    :param V_ref: any arbitrary voltage
    :return: the normalized meshes from input V1, V2 and P
    """
    # scalars
    # V_ref = any arbitrary voltage
    Z_ref = 2 * np.pi * fs_nom * L_s
    P_ref = V_ref ** 2 / Z_ref
    # arrays
    V1n = mesh_V1 / V_ref
    V2n = n * mesh_V2 / V_ref
    Pn = mesh_P / P_ref

    # debug(V1n, V2n, Pn, sep='\n')
    return V1n, V2n, Pn


def _Pn_max(V1n: np.ndarray, V2n: np.ndarray) -> np.ndarray:
    """
    Calculate Pn_max, which is the limit for SPS/CPM and therefore the absolute max for the DAB!

    :param V1n: normalized input voltage (voltage on side 1)
    :param V2n: normalized output voltage (voltage on side 2)
    :return: Pn_max
    """
    Pn_max = (np.pi * V1n * V2n) / 4

    # debug(Pn_max, sep='\n')
    return Pn_max


def _limit_Pn(Pn: np.ndarray, Pn_max: np.ndarray) -> np.ndarray:
    """
    Replace Pn values beyond Pn_max with values from Pn_max

    :param Pn:
    :param Pn_max:
    :return:
    """
    Pn_limit = np.clip(Pn, -Pn_max, Pn_max)

    if not np.all(np.equal(Pn, Pn_limit)):
        warning('Pn was limited! False elements show the limited elements.', np.equal(Pn, Pn_limit))

    # debug(Pn_limit, sep='\n')
    return Pn_limit


def _calc_TCM(Van: np.ndarray, Vbn: np.ndarray, Pn: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    TCM (Triangle Conduction Mode) Modulation calculation, which will return phi, tau1 and tau2
    It will not be checked if Pn <= Pn_tcmmax

    :param Van:
    :param Vbn:
    :param Pn:
    :return:
    """
    # TCM: calculate phi, Da and Db with (21)

    # interim value for what goes into sqrt
    _isqrt = (Vbn - Van) / (2 * np.power(Van, 2) * Vbn) * np.abs(Pn) / np.pi
    _isqrt[_isqrt < 0] = np.nan
    phi = np.pi * np.sign(Pn) * np.sqrt(_isqrt)
    # phi[_bsqrt_elem_not_negative] = np.pi * np.sign(Pn)[_bsqrt_elem_not_negative] * np.sqrt(_isqrt[_bsqrt_elem_not_negative])

    Da = np.abs(phi) / np.pi * Vbn / (Vbn - Van)

    Db = np.abs(phi) / np.pi * Van / (Vbn - Van)

    # debug(phi, Da, Db, sep='\n')
    return phi, Da, Db


def _calc_OTM(Van: np.ndarray, Vbn: np.ndarray, Pn: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    OTM (Optimal Transition Mode) Modulation calculation, which will return phi, tau1 and tau2
    It will not be checked if Pn_tcmmax < Pn <= Pn_optmax

    :param Van:
    :param Vbn:
    :param Pn:
    :return:
    """
    # OTM: calculate phi, Da and Db with (23) and (24)

    # (23) OTM Da and Db
    # Formula is taken *as-is* from the Paper with all its interim values

    # Some more interim values that may improve performance
    # f1 = np.power(Van, 2) + np.power(Van, 2)
    # f2 = np.abs(Pn) / np.pi
    # This returns some very odd results!

    e1 = - (2 * np.power(Van, 2) + np.power(Van, 2)) / (np.power(Van, 2) + np.power(Van, 2))

    e2 = (np.power(Van, 3) * Vbn + np.abs(Pn) / np.pi * (np.power(Van, 2) + np.power(Van, 2))) / \
         (np.power(Van, 3) * Vbn + Van * np.power(Van, 3))

    e3 = 8 * np.power(Van, 7) * np.power(Van, 5) - 64 * np.power(np.abs(Pn) / np.pi, 3) * \
         np.power((np.power(Van, 2) + np.power(Van, 2)), 3) - \
         np.abs(Pn) / np.pi * np.power(Van, 4) * np.power(Van, 2) * (4 * np.power(Van, 2) + np.power(Van, 2)) * \
         (4 * np.power(Van, 2) + 13 * np.power(Van, 2)) + \
         16 * np.power(np.abs(Pn) / np.pi, 2) * Van * np.power((np.power(Van, 2) + np.power(Van, 2)), 2) * \
         (4 * np.power(Van, 2) * Vbn + np.power(Van, 3))
    # Mask values <0 with NaN so that we can do sqrt(e3) in the next step
    e3[e3 < 0] = np.nan

    e4 = 8 * np.power(Van, 9) * np.power(Van, 3) - 8 * np.power(np.abs(Pn) / np.pi, 3) * \
         (8 * np.power(Van, 2) - np.power(Van, 2)) * np.power((np.power(Van, 2) + np.power(Van, 2)), 2) - \
         12 * np.abs(Pn) / np.pi * np.power(Van, 6) * np.power(Van, 2) * (4 * np.power(Van, 2) + np.power(Van, 2)) + \
         3 * np.power(np.abs(Pn) / np.pi, 2) * np.power(Van, 3) * Vbn * (4 * np.power(Van, 2) + np.power(Van, 2)) * \
         (8 * np.power(Van, 2) + 5 * np.power(Van, 2)) + \
         np.power((3 * np.abs(Pn) / np.pi), (3 / 2)) * Van * np.power(Van, 2) * np.sqrt(e3)

    e5 = (2 * np.power(Van, 6) * np.power(Van, 2) + 2 * np.abs(Pn) / np.pi * (4 * np.power(Van, 2) + np.power(Van, 2)) *
          (np.abs(Pn) / np.pi * (np.power(Van, 2) + np.power(Van, 2)) - np.power(Van, 3) * Vbn)) * \
         1 / (3 * Van * Vbn * (np.power(Van, 2) + np.power(Van, 2)) * np.power(e4, (1 / 3)))

    e6 = (4 * (np.power(Van, 3) * np.power(Van, 2) + 2 * np.power(Van, 5)) + 4 *
          np.abs(Pn) / np.pi * (np.power(Van, 2) * Vbn + np.power(Van, 3))) / \
         (Van * np.power((np.power(Van, 2) + np.power(Van, 2)), 2))

    e7 = np.power(e4, (1 / 3)) / (6 * np.power(Van, 3) * Vbn + 6 * Van * np.power(Van, 3)) + \
         np.power(e1, 2) / 4 - (2 * e2) / 3 + e5
    # Mask values <0 with NaN so that we can do sqrt(e7) in the next step
    e7[e7 < 0] = np.nan

    e8 = 1 / 4 * ((- np.power(e1, 3) - e6) / np.sqrt(e7) + 3 * np.power(e1, 2) - 8 * e2 - 4 * e7)
    # Mask values <0 with NaN so that we can do sqrt(e8) in the next step
    e8[e8 < 0] = np.nan

    # The resulting formulas:

    Db = 1 / 4 * (2 * np.sqrt(e7) - 2 * np.sqrt(e8) - e1)

    Da = np.full_like(Db, 1 / 2)

    # (24) OTM phi
    phi = np.pi * np.sign(Pn) * \
          (1 / 2 - np.sqrt(Da * (1 - Da) + Db * (1 - Db) - 1 / 4 - np.abs(Pn) / np.pi * 1 / (Van * Vbn)))

    # debug(phi, Da, Db, sep='\n')
    return phi, Da, Db


def _calc_CPM(Van: np.ndarray, Vbn: np.ndarray, Pn: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    CPM (Conventional Phaseshift Modulation) (egal to SPS Modulation) calculation, which will return phi, tau1 and tau2
    It will not be checked if abs(Pn) > Pn_optmax and abs(Pn) <= Pn_max

    :param Van:
    :param Vbn:
    :param Pn:
    :return:
    """
    # CPM/SPS: calculate phi, Da and Db with (26)
    phi = np.pi * np.sign(Pn) * (1 / 2 - np.sqrt(1 / 4 - np.abs(Pn) / np.pi * 1 / (Van * Vbn)))

    Da = np.full_like(phi, 1 / 2)

    Db = np.full_like(phi, 1 / 2)

    # debug(phi, Da, Db, sep='\n')
    return phi, Da, Db


def convert_phiM_to_phiF(phi, tau1, tau2):
    """
    Convert phi from Middle-Pulse alignment to First-Falling-Edge alignment.
    Middle-Pulse alignment:         _____|--M--|
    First-Falling-Edge alignment:   _____|-----F
    :param phi:
    :param tau1:
    :param tau2:
    :return: phi
    """
    phi = phi - (tau1 - tau2) / 2
    # Fix tiny values like e-16 to useful resolution
    phi = np.round(phi, decimals=8)
    return phi


def convert_phiF_to_phiM(phi, tau1, tau2):
    """
    Convert phi from First-Falling-Edge alignment to Middle-Pulse alignment.
    Middle-Pulse alignment:         _____|--M--|
    First-Falling-Edge alignment:   _____|-----F
    :param phi:
    :param tau1:
    :param tau2:
    :return: phi
    """
    phi = phi + (tau1 - tau2) / 2
    # Fix tiny values like e-16 to useful resolution
    phi = np.round(phi, decimals=8)
    return phi


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module MCL ...")

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
    dab_results.append_result_dict(da_mod, name_pre='mod_mcl_')

    # print("mod_mcl_phi:", dab_results.mod_mcl_phi, sep='\n')
    # print("mod_mcl_tau1:", dab_results.mod_mcl_tau1, sep='\n')
    # print("mod_mcl_tau2:", dab_results.mod_mcl_tau2, sep='\n')
    # print("mod_mcl_phi[0,0,0]", type(dab_results.mod_mcl_phi[0, 0, 0]))
    # print("mod_mcl_tau1[0,0,0]", type(dab_results.mod_mcl_tau1[0, 0, 0]))
    # print("mod_mcl_tau2[0,0,0]", type(dab_results.mod_mcl_tau2[0, 0, 0]))

    info("\nStart Plotting\n")
    import plot_dab

    v1_middle = int(np.shape(dab_results.mesh_P)[1] / 2)

    plt = plot_dab.Plot_DAB()
    # Plot all modulation angles
    # plt.new_fig(nrows=1, ncols=3, tab_title='DAB Modulation Angles')
    plt.plot_modulation(dab_results.mesh_P[:, v1_middle, :],
                        dab_results.mesh_V2[:, v1_middle, :],
                        dab_results.mod_mcl_phi[:, v1_middle, :],
                        dab_results.mod_mcl_tau1[:, v1_middle, :],
                        dab_results.mod_mcl_tau2[:, v1_middle, :],
                        mask1=dab_results.mod_mcl_mask_tcm[:, v1_middle, :],
                        mask2=dab_results.mod_mcl_mask_cpm[:, v1_middle, :],
                        tab_title='DAB Modulation Angles')

    # plt.new_fig(nrows=1, ncols=3, tab_title='MCL Modulation Angles')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                      dab_results.mesh_V2[:, v1_middle, :],
    #                      dab_results.mod_mcl_phi[:, v1_middle, :] / np.pi * 180,
    #                      ax=plt.figs_axes[-1][1][0],
    #                      xlabel=r'$P / \mathrm{W}$', ylabel=r'$U_2 / \mathrm{V}$', title=r'$\varphi / \mathrm{rad}$')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                      dab_results.mesh_V2[:, v1_middle, :],
    #                      dab_results.mod_mcl_tau1[:, v1_middle, :] / np.pi * 180,
    #                      ax=plt.figs_axes[-1][1][1],
    #                      xlabel=r'$P / \mathrm{W}$', ylabel=r'$U_2 / \mathrm{V}$', title=r'$\tau_1 / \mathrm{rad}$')
    # plt.subplot_contourf(dab_results.mesh_P[:, v1_middle, :],
    #                      dab_results.mesh_V2[:, v1_middle, :],
    #                      dab_results.mod_mcl_tau2[:, v1_middle, :] / np.pi * 180,
    #                      ax=plt.figs_axes[-1][1][2],
    #                      xlabel=r'$P / \mathrm{W}$', ylabel=r'$U_2 / \mathrm{V}$', title=r'$\tau_2 / \mathrm{rad}$')

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

    plt.show()
