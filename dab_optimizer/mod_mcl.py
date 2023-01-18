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
    Pn_max = _limit_Pn(Pn, Pn_max)

    # Determine Van and Vbn with (20)
    # input value mapping
    # TODO ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    if np.less_equal(V1n, V2n):
        Van = V1n
        Vbn = V2n
    else:
        if np.greater(V1n, V2n):
            Van = V2n
            Vbn = V1n
        else:
            error('Neither is V1n <= V2n or V1n > V2n, therefore there must be an overlap in V1n and V2n!')

    # Calculate Pn_tcm,max with (22). Maximum power for TCM!
    # TODO maybe check if DAB P_max it higher
    Pn_tcmmax = np.pi / 2 * (np.power(Van, 2) * (Vbn - Van)) / Vbn

    # if abs(Pn) <= Pn_tcmmax: calc TCM else: do the rest
    # TODO how to calculate thins partially?

    # TCM: calculate phi, Da and Db with (21)
    phi_tcm, da_tcm, db_tcm = _calc_TCM(Van, Vbn, Pn)

    # OTM: calculate phi, Da and Db with (23) and (24)
    phi_otm, da_otm, db_otm = _calc_OTM(Van, Vbn, Pn)

    # CPM/SPS: calculate phi, Da and Db with (26)
    phi_cpm, da_cpm, db_cpm = _calc_CPM(Van, Vbn, Pn)






    # TODO ********** ONLY for DEBUG start **********
    phi = phi_tcm
    Da = da_tcm
    Db = da_tcm
    # TODO ********** ONLY for DEBUG end **********

    # output value mapping
    if np.less_equal(V1n, V2n):
        D1 = Da
        D2 = Db
    else:
        if np.greater(V1n, V2n):
            D2 = Da
            D1 = Db
        else:
            error('Neither is V1n <= V2n or V1n > V2n, therefore there must be an overlap in V1n and V2n!')

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

    debug(V1n, V2n, Pn, sep='\n')
    return V1n, V2n, Pn

def _Pn_max(V1n: np.ndarray, V2n: np.ndarray) -> np.ndarray:
    """
    Calculate Pn_max, which is the limit for SPS/CPM and therefore the absolute max for the DAB!

    :param V1n: normalized input voltage (voltage on side 1)
    :param V2n: normalized output voltage (voltage on side 2)
    :return: Pn_max
    """
    Pn_max = (np.pi * V1n * V2n) / 4

    debug(Pn_max, sep='\n')
    return Pn_max

def _limit_Pn(Pn: np.ndarray, Pn_max: np.ndarray) -> np.ndarray:
    """
    Replace Pn values beyond Pn_max with values from Pn_max

    :param Pn:
    :param Pn_max:
    :return:
    """
    Pn_limit = np.clip(Pn, -Pn_max, Pn_max)

    debug(Pn_limit, sep='\n')
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
    phi = np.pi * np.sgn(Pn) * np.sqrt((Vbn - Van) / (2 * np.power(Van, 2) * Vbn) * np.abs(Pn) / np.pi)

    Da = np.abs(phi) / np.pi * Vbn / (Vbn - Van)

    Db = np.abs(phi) / np.pi * Van / (Vbn - Van)

    debug(phi, Da, Db, sep='\n')
    return phi, Da, Db

def _calc_OTM(Van: np.ndarray, Vbn: np.ndarray, Pn: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    TCM (Triangle Conduction Mode) Modulation calculation, which will return phi, tau1 and tau2
    It will not be checked if Pn <= Pn_tcmmax

    :param Van:
    :param Vbn:
    :param Pn:
    :return:
    """
    #TODO
    # TCM: calculate phi, Da and Db with (21)
    phi = np.pi * np.sgn(Pn) * np.sqrt((Vbn - Van) / (2 * np.power(Van, 2) * Vbn) * np.abs(Pn) / np.pi)

    Da = np.abs(phi) / np.pi * Vbn / (Vbn - Van)

    Db = np.abs(phi) / np.pi * Van / (Vbn - Van)

    debug(phi, Da, Db, sep='\n')
    return phi, Da, Db

def _calc_CPM(Van: np.ndarray, Vbn: np.ndarray, Pn: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    TCM (Triangle Conduction Mode) Modulation calculation, which will return phi, tau1 and tau2
    It will not be checked if Pn <= Pn_tcmmax

    :param Van:
    :param Vbn:
    :param Pn:
    :return:
    """
    #TODO
    # TCM: calculate phi, Da and Db with (21)
    phi = np.pi * np.sgn(Pn) * np.sqrt((Vbn - Van) / (2 * np.power(Van, 2) * Vbn) * np.abs(Pn) / np.pi)

    Da = np.abs(phi) / np.pi * Vbn / (Vbn - Van)

    Db = np.abs(phi) / np.pi * Van / (Vbn - Van)

    debug(phi, Da, Db, sep='\n')
    return phi, Da, Db



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
