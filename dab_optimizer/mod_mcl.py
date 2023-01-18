#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

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


@timeit
def calc_modulation(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P):
    """
    MCL (Minimum Conduction Loss) Modulation calculation, which will return phi, tau1 and tau2

    :param n: Transformer turns ratio
    :param L_s: DAB converter inductance.
    :param fs_nom: Switching frequency
    :param mesh_V1: input voltage (voltage on side 1)
    :param mesh_V2: output voltage (voltage on side 2)
    :param mesh_P: DAB power (assuming a lossless DAB)
    :return: phi, tau1, tau2
    """

    # Reference voltage, any arbitrary voltage!
    V_ref = 100

    # Calculate normalized voltage and power values V1n, V2n, Pn
    V1n, V2n, Pn = _normalize_input_arrays(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P, V_ref)

    # Calculate Pn_max with (14)
    # TODO maybe check if inductor L_s is too big for the power P_max
    Pn_max = _Pn_max(V1n, V2n)

    # Limit Pn to +/- Pn_max
    Pn_no_lim = Pn # saved just in case...
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
    # all the modulations are calculated first even for useless areas and we decide later which part is useful.
    # This should be faster and easier.

    # TCM: calculate phi, Da and Db with (21)
    phi_tcm, da_tcm, db_tcm = _calc_TCM(Van, Vbn, Pn)

    # OTM: calculate phi, Da and Db with (23) and (24)
    phi_otm, da_otm, db_otm = _calc_OTM(Van, Vbn, Pn)

    # CPM/SPS: calculate phi, Da and Db with (26)
    phi_cpm, da_cpm, db_cpm = _calc_CPM(Van, Vbn, Pn)

    # if abs(Pn) <= Pn_tcmmax: use TCM
    #_tcm_mask = np.full_like(Pn, np.nan)
    _tcm_mask = np.less_equal(np.abs(Pn), Pn_tcmmax)
    # debug('TCM MASK\n', _tcm_mask)

    # Calculate Pn_optmax with (25)
    # OTM for Pn_tcmmax < abs(Pn) <= Pn_optmax
    _otm_mask = np.less_equal(db_otm, 1/2)
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

    # use TCM: if abs(Pn) <= Pn_tcmmax
    phi[_tcm_mask] = phi_tcm[_tcm_mask]
    Da[_tcm_mask] = da_tcm[_tcm_mask]
    Db[_tcm_mask] = db_tcm[_tcm_mask]

    # use OTM: if Pn_tcmmax < abs(Pn) <= Pn_optmax
    phi[_otm_mask] = phi_otm[_otm_mask]
    Da[_otm_mask] = da_otm[_otm_mask]
    Db[_otm_mask] = db_otm[_otm_mask]

    # use CPM: if abs(Pn) > Pn_optmax
    phi[_cpm_mask] = phi_cpm[_cpm_mask]
    Da[_cpm_mask] = da_cpm[_cpm_mask]
    Db[_cpm_mask] = db_cpm[_cpm_mask]

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
    tau1 = D1 * np.pi
    tau2 = D2 * np.pi

    return phi, tau1, tau2

def _normalize_input_arrays(n, L_s, fs_nom, mesh_V1, mesh_V2, mesh_P, V_ref: float = 100):
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

@timeit
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
    #phi[_bsqrt_elem_not_negative] = np.pi * np.sign(Pn)[_bsqrt_elem_not_negative] * np.sqrt(_isqrt[_bsqrt_elem_not_negative])

    Da = np.abs(phi) / np.pi * Vbn / (Vbn - Van)

    Db = np.abs(phi) / np.pi * Van / (Vbn - Van)

    # debug(phi, Da, Db, sep='\n')
    return phi, Da, Db

@timeit
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

    # TODO Some more interim values that may improve performance
    # f1 = np.power(Van, 2) + np.power(Van, 2)
    # f2 = np.abs(Pn) / np.pi

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

    Da = np.full_like(Db, 1/2)

    # (24) OTM phi
    phi = np.pi * np.sign(Pn) * \
          (1 / 2 - np.sqrt(Da * (1 - Da) + Db * (1 - Db) - 1 / 4 - np.abs(Pn) / np.pi * 1 / (Van * Vbn)))

    # debug(phi, Da, Db, sep='\n')
    return phi, Da, Db

@timeit
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

    Da = np.full_like(phi, 1/2)

    Db = np.full_like(phi, 1/2)

    # debug(phi, Da, Db, sep='\n')
    return phi, Da, Db



# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module CPM ...")

    Dab_Specs = ds.DAB_Specification()
    Dab_Specs.V1_nom = 700
    Dab_Specs.V1_min = 600
    Dab_Specs.V1_max = 800
    Dab_Specs.V1_step = 21
    Dab_Specs.V2_nom = 235
    Dab_Specs.V2_min = 175
    Dab_Specs.V2_max = 295
    Dab_Specs.V2_step = 25
    Dab_Specs.P_min = 400
    Dab_Specs.P_max = 2200
    Dab_Specs.P_nom = 2000
    Dab_Specs.P_step = 19
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

    # print("mod_phi:", Dab_Results.mod_phi, sep='\n')
    # print("mod_tau1:", Dab_Results.mod_tau1, sep='\n')
    # print("mod_tau2:", Dab_Results.mod_tau2, sep='\n')
    # print("mod_phi[0,0,0]", type(Dab_Results.mod_phi[0, 0, 0]))
    # print("mod_tau1[0,0,0]", type(Dab_Results.mod_tau1[0, 0, 0]))
    # print("mod_tau2[0,0,0]", type(Dab_Results.mod_tau2[0, 0, 0]))

    info("\nStart Plotting\n")
    import plot_dab
    Plot_Dab = plot_dab.Plot_DAB()
    # Plot all modulation angles
    Plot_Dab.new_fig(nrows=1, ncols=3, tab_title='DAB Modulation Angles')
    Plot_Dab.plot_modulation(Plot_Dab.figs_axes[-1],
                      Dab_Results.mesh_P[:, 1, :],
                      Dab_Results.mesh_V2[:, 1, :],
                      Dab_Results.mod_phi[:, 1, :],
                      Dab_Results.mod_tau1[:, 1, :],
                      Dab_Results.mod_tau2[:, 1, :])
    Plot_Dab.show()
