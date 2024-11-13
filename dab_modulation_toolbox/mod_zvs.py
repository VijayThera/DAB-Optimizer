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

import dab_datasets as ds
from debug_tools import *

# The dict keys this modulation will return
MOD_KEYS = ['phi', 'tau1', 'tau2', 'mask_zvs', 'mask_Im2', 'mask_IIm2',
            'mask_IIIm1', 'zvs_coverage', 'zvs_coverage_notnan']


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

    # Create empty meshes
    phi = np.full_like(V1, np.nan)
    tau1 = np.full_like(V1, np.nan)
    tau2 = np.full_like(V1, np.nan)
    zvs = np.full_like(V1, np.nan)
    _Im2_mask = np.full_like(V1, False)
    _IIm2_mask = np.full_like(V1, False)
    _IIIm1_mask = np.full_like(V1, False)

    ## Precalculate all required values
    # Transform Lc2 to side 1
    Lc2_ = Lc2 * n ** 2
    # Transform V2 to side 1
    V2_ = V2 * n
    # For negative P we have to recalculate phi at the end
    _negative_power_mask = np.less(P, 0)
    I1 = np.abs(P) / V1
    # Convert fs into omega_s
    ws = 2 * np.pi * fs
    # parasitic capacitance with copper blocks, TIM, and heatsink
    C_Par1 = 42e-12 # 6e-12
    C_Par2 = 42e-12 # 6e-12
    # 20% higher for safety margin
    C_total_1 = 1.2 * (Coss1 + C_Par1)
    C_total_2 = 1.2 * (Coss2 + C_Par2)
    # Calculate required Q for each voltage
    Q_AB_req1 = _integrate_Coss(C_total_1 * 2, V1)
    Q_AB_req2 = _integrate_Coss(C_total_2 * 2, V2)

    # FIXME HACK for testing V1, V2 interchangeability
    # _V1 = V1
    # V1 = V2_
    # V2_ = V1

    ## Calculate the Modulations
    # ***** Change in contrast to paper *****
    # Instead of fist checking the limits for each modulation and only calculate each mod. partly,
    # all the modulations are calculated first even for useless areas, and we decide later which part is useful.
    # This should be faster and easier.

    # Int. I (mode 2): calculate phi, tau1 and tau2
    phi_I, tau1_I, tau2_I = _calc_interval_I(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

    # Int. II (mode 2): calculate phi, tau1 and tau2
    phi_II, tau1_II, tau2_II = _calc_interval_II(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

    # Int. III (mode 1): calculate phi, tau1 and tau2
    phi_III, tau1_III, tau2_III, tau2_III_g_pi_mask = _calc_interval_III(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

    ## Decision Logic
    # Int. I (mode 2):
    # if phi <= 0:
    _phi_I_leq_zero_mask = np.less_equal(phi_I, 0)
    # debug('_phi_I_leq_zero_mask', _phi_I_leq_zero_mask)
    # if tau1 <= pi:
    _tau1_I_leq_pi_mask = np.less_equal(tau1_I, np.pi)
    # debug('_tau1_I_leq_pi_mask', _tau1_I_leq_pi_mask)
    # if phi > 0:
    _phi_I_g_zero_mask = np.greater(phi_I, 0)
    # debug('_phi_I_g_zero_mask', _phi_I_g_zero_mask)
    _Im2_mask = np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)

    # Int. II (mode 2):
    # if tau1 <= pi:
    _tau1_II_leq_pi_mask = np.less_equal(tau1_II, np.pi)
    # debug('_tau1_II_leq_pi_mask', _tau1_II_leq_pi_mask)
    # if tau1 > pi:
    _tau1_II_g_pi_mask = np.greater(tau1_II, np.pi)
    # debug('_tau1_II_g_pi_mask', _tau1_II_g_pi_mask)

    # Int. III (mode 1):
    # if tau2 <= pi:
    _tau2_III_leq_pi_mask = np.less_equal(tau2_III, np.pi)
    # debug('_tau2_III_leq_pi_mask', _tau2_III_leq_pi_mask)

    # Walking backwards to the algorithm so make sure that earlier modulation parameters have higher priority

    # update modulation parameters for tau2 = pi, last step
    phi[tau2_III_g_pi_mask] = phi_III[tau2_III_g_pi_mask]
    tau1[tau2_III_g_pi_mask] = tau1_III[tau2_III_g_pi_mask]
    tau2[tau2_III_g_pi_mask] = tau2_III[tau2_III_g_pi_mask]
    zvs[tau2_III_g_pi_mask] = True

    _IIIm1_mask = np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)
    phi[_IIIm1_mask] = phi_III[_IIIm1_mask]
    tau1[_IIIm1_mask] = tau1_III[_IIIm1_mask]
    tau2[_IIIm1_mask] = tau2_III[_IIIm1_mask]
    zvs[_IIIm1_mask] = True

    _IIm2_mask = np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)
    phi[_IIm2_mask] = phi_II[_IIm2_mask]
    tau1[_IIm2_mask] = tau1_II[_IIm2_mask]
    tau2[_IIm2_mask] = tau2_II[_IIm2_mask]
    zvs[_IIm2_mask] = True

    # Int. I (mode 2): ZVS is analytically IMPOSSIBLE!
    zvs[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))] = False
    phi[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))] = phi_I[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))]
    tau1[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))] = tau1_I[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))]
    tau2[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))] = tau2_I[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))]

    phi[_Im2_mask] = phi_I[_Im2_mask]
    tau1[_Im2_mask] = tau1_I[_Im2_mask]
    tau2[_Im2_mask] = tau2_I[_Im2_mask]
    zvs[_Im2_mask] = True

    ## Recalculate phi for negative power
    phi_nP = - (tau1 + phi - tau2)
    phi[_negative_power_mask] = phi_nP[_negative_power_mask]

    # Init return dict
    da_mod_results = dict()
    # Save the results in the dict
    # da_mod_results[MOD_KEYS[0]] = phi
    # Convert phi because the math from the paper uses Middle-Pulse alignment, but we use First-Falling-Edge alignment!
    da_mod_results[MOD_KEYS[0]] = phi
    da_mod_results[MOD_KEYS[1]] = tau1
    da_mod_results[MOD_KEYS[2]] = tau2
    da_mod_results[MOD_KEYS[3]] = zvs
    da_mod_results[MOD_KEYS[4]] = _Im2_mask
    da_mod_results[MOD_KEYS[5]] = _IIm2_mask
    da_mod_results[MOD_KEYS[6]] = np.bitwise_or(_IIIm1_mask, tau2_III_g_pi_mask)

    # ZVS coverage based on calculation: Percentage ZVS based on all points (full operating range)
    da_mod_results[MOD_KEYS[7]] = np.count_nonzero(zvs) / np.size(zvs)
    # ZVS coverage based on calculation: Percentage ZVS based on all points where the converter can be operated (not full operating range)
    da_mod_results[MOD_KEYS[8]] = np.count_nonzero(zvs[~np.isnan(tau1)]) / np.size(zvs[~np.isnan(tau1)])

    # debug(da_mod_results)
    return da_mod_results


def _integrate_Coss(coss: np.ndarray, V: np.ndarray) -> np.ndarray:
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
    # get an qoss vector that has the resolution 1V from 0 to V_max
    v_vec = np.arange(coss.shape[0])
    # get an qoss vector that fits the mesh_V scale
    # v_vec = np.linspace(V_min, V_max, int(V_step))
    qoss = coss_int(v_vec)

    # Calculate a qoss mesh that is like the V mesh
    # Each element in V gets its q(v) value
    def meshing_q(v):
        return np.interp(v, v_vec, qoss)

    q_meshgrid = np.vectorize(meshing_q)
    qoss_mesh = q_meshgrid(V)

    return qoss_mesh


def _calc_interval_I(n, Ls, Lc1, Lc2_, ws: np.ndarray | int | float, Q_AB_req1: np.ndarray, Q_AB_req2: np.ndarray,
                     V1: np.ndarray, V2_: np.ndarray, I1: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 2 Modulation (interval I) calculation, which will return phi, tau1 and tau2
    """
    ## Predefined Terms
    e1 = V2_ * Q_AB_req2 * ws

    e2 = n * V1 * np.pi * I1

    # FIXME e3 gets negative for all values n*V2 < V1, why? Formula is checked against PhD.
    # TODO Maybe Ls is too small? Is that even possible? Error in Formula?
    e3 = n * (V2_ * (Lc2_ + Ls) - V1 * Lc2_)
    # if np.any(np.less(e3, 0)):
    # warning('Something is wrong. Formula e3 is negative and it should not!')
    # warning('Please check your DAB Params, probably you must check n or iterate L, Lc1, Lc2.')
    # warning(V2_, Lc2_, V1, Ls)

    e4 = 2 * n * np.sqrt(Q_AB_req1 * Ls * np.power(ws, 2) * V1 * Lc1 * (Lc1 + Ls))

    e5 = Ls * Lc2_ * ws * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    # debug('e1',e1,'e2',e2,'e3',e3,'e4',e4,'e5',e5)

    ## Solution for interval I (mode 2)
    tau1 = (np.sqrt(2) * (Lc1 * np.sqrt(V2_ * e3 * e5) + e4 * e3 * 1 / n)) / (V1 * e3 * (Lc1 + Ls))

    tau2 = np.sqrt((2 * e5) / (V2_ * e3))

    phi = (tau2 - tau1) / 2 + (I1 * ws * Ls * np.pi) / (tau2 * V2_)

    # debug(phi, tau1, tau2)
    return phi, tau1, tau2


def _calc_interval_II(n, Ls, Lc1, Lc2_, ws: np.ndarray | int | float, Q_AB_req1: np.ndarray, Q_AB_req2: np.ndarray,
                      V1: np.ndarray, V2_: np.ndarray, I1: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 2 Modulation (interval II) calculation, which will return phi, tau1 and tau2
    """
    ## Predefined Terms
    e1 = V2_ * Q_AB_req2 * ws

    e2 = n * V1 * np.pi * I1

    e3 = n * (V2_ * (Lc2_ + Ls) - V1 * Lc2_)

    e4 = 2 * n * np.sqrt(Q_AB_req1 * Ls * np.power(ws, 2) * V1 * Lc1 * (Lc1 + Ls))

    e5 = Ls * Lc2_ * ws * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    ## Solution for interval II (mode 2)
    tau1 = (np.sqrt(2) * (e5 + ws * Ls * Lc2_ * e2 * (V2_ / V1 * (Ls / Lc2_ + 1) - 1))) / (np.sqrt(V2_ * e3 * e5))

    tau2 = np.sqrt((2 * e5) / (V2_ * e3))

    phi = np.full_like(V1, 0)

    # debug(phi, tau1, tau2)
    return phi, tau1, tau2


def _calc_interval_III(n, Ls, Lc1, Lc2_, ws: np.ndarray | int | float, Q_AB_req1: np.ndarray, Q_AB_req2: np.ndarray,
                       V1: np.ndarray, V2_: np.ndarray, I1: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Mode 1 Modulation (interval III) calculation, which will return phi, tau1 and tau2
    """
    ## Predefined Terms
    e1 = V2_ * Q_AB_req2 * ws

    e2 = n * V1 * np.pi * I1

    e3 = n * (V2_ * (Lc2_ + Ls) - V1 * Lc2_)

    e4 = 2 * n * np.sqrt(Q_AB_req1 * Ls * np.power(ws, 2) * V1 * Lc1 * (Lc1 + Ls))

    e5 = Ls * Lc2_ * ws * (e2 + 2 * e1 + 2 * np.sqrt(e1 * (e1 + e2)))

    ## Solution for interval III (mode 1)
    tau1 = np.full_like(V1, np.pi)

    tau2 = np.sqrt((2 * e5) / (V2_ * e3))

    phi = (- tau1 + tau2 + np.pi) / 2 - np.sqrt(
        (- np.power((tau2 - np.pi), 2) + tau1 * (2 * np.pi - tau1)) / 4 - (I1 * ws * Ls * np.pi) / V2_)

    ## Check if tau2 > pi: Set tau2 = pi and recalculate phi for these points

    # if tau2 > pi:
    _tau2_III_g_pi_mask = np.greater(tau2, np.pi)
    # debug('_tau2_III_g_pi_mask', _tau2_III_g_pi_mask)
    tau2_ = np.full_like(V1, np.pi)
    phi_ = (- tau1 + tau2_ + np.pi) / 2 - np.sqrt(
        (- np.power((tau2_ - np.pi), 2) + tau1 * (2 * np.pi - tau1)) / 4 - (I1 * ws * Ls * np.pi) / V2_)
    tau2[_tau2_III_g_pi_mask] = tau2_[_tau2_III_g_pi_mask]
    phi[_tau2_III_g_pi_mask] = phi_[_tau2_III_g_pi_mask]

    # debug(phi, tau1, tau2)
    return phi, tau1, tau2


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module ZVS ...")

    ## Normal DAB
    # Set the basic DAB Specification
    dab = ds.DAB_Data()
    dab.V1_nom = 700
    dab.V1_min = 690
    dab.V1_max = 710
    dab.V1_step = 20
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = 25 * 3
    dab.P_min = 0
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 19 * 3
    dab.n = 3.974
    dab.Ls = 137.3e-6
    dab.Lc1 = 619e-6
    dab.Lc2 = 608.9e-6 / (dab.n ** 2)
    dab.Lm = 595e-6
    dab.fs = 200000
    # Generate meshes
    dab.gen_meshes()

    # ## Reversed DAB
    # # Set the basic DAB Specification
    # dab = ds.DAB_Data()
    # dab.V2_nom = 700
    # dab.V2_min = 600
    # dab.V2_max = 800
    # dab.V2_step = 3
    # dab.V1_nom = 235
    # dab.V1_min = 175
    # dab.V1_max = 295
    # dab.V1_step = 25 * 3
    # #dab.V2_step = 4
    # dab.P_min = -2200
    # dab.P_max = 2200
    # dab.P_nom = 2000
    # dab.P_step = 19 * 3
    # #dab.P_step = 5
    # dab.n = 1 / 2.99
    # dab.Ls = 83e-6 * dab.n ** 2
    # dab.Lm = 595e-6 * dab.n ** 2
    # #dab.Lc1 = 25.62e-3
    # #dab.Lc1 = 800e-6
    # # Assumption for tests
    # dab.Lc1 = 611e-6 * dab.n ** 2
    # dab.Lc2 = 611e-6 * dab.n ** 2
    # #dab.Lc2 = 25e-3 * dab.n ** 2
    # dab.fs = 200000
    # # Generate meshes
    # dab.gen_meshes()

    # ## DAB Everts
    # # Set the basic DAB Specification
    # dab = ds.DAB_Data()
    # dab.V1_nom = 250
    # #dab.V1_min = 30
    # dab.V1_min = 125
    # dab.V1_max = 325
    # dab.V1_step = math.floor((dab.V1_max - dab.V1_min) / 10 + 1)  # 10V resolution gives 21 steps
    # # dab.V1_step = math.floor((dab.V1_max - dab.V1_min) / 10 + 1)
    # # dab.V1_step = 1
    # dab.V2_nom = 400
    # dab.V2_min = 370
    # dab.V2_max = 470
    # dab.V2_step = math.floor((dab.V2_max - dab.V2_min) / 10 + 1)  # 5V resolution gives 25 steps
    # # dab.V2_step = math.floor((dab.V2_max - dab.V2_min) / 20 + 1)
    # # dab.V2_step = 4
    # #dab.P_min = -3700
    # dab.P_min = -3700
    # dab.P_max = 3700
    # dab.P_nom = 2000
    # dab.P_step = math.floor((dab.P_max - dab.P_min) / 100 + 1)  # 100W resolution gives 19 steps
    # # dab.P_step = math.floor((dab.P_max - dab.P_min) / 300 + 1)
    # # dab.P_step = 5
    # dab.n = 1
    # dab.Ls = 13e-6
    # dab.Lc1 = 62.1e-6
    # dab.Lc2 = 62.1e-6
    # dab.fs = 120e3
    # # Generate meshes
    # dab.gen_meshes()

    # Import Coss curves
    # mosfet1 = 'C3M0120100J'
    mosfet1 = 'C3M0065100J'
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    csv_file = os.path.join(csv_file, 'Coss_files', f'Coss_{mosfet1}.csv')
    dab.import_Coss(csv_file, mosfet1)
    # mosfet2 = 'C3M0120100J'
    mosfet2 = 'C3M0060065J'
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    csv_file = os.path.join(csv_file, 'Coss_files', f'Coss_{mosfet2}.csv')
    dab.import_Coss(csv_file, mosfet2)

    # Modulation Calculation
    # ZVS Modulation
    # calc_modulation(n, Ls, Lc1, Lc2, fs: np.ndarray | int | float, Coss1: np.ndarray, Coss2: np.ndarray,
    #                 V1: np.ndarray, V2: np.ndarray, P: np.ndarray)
    da_mod = calc_modulation(dab.n,
                             dab.Ls,
                             dab.Lc1,
                             dab.Lc2,
                             dab.fs,
                             dab['coss_' + mosfet1],
                             dab['coss_' + mosfet2],
                             dab.mesh_V1,
                             dab.mesh_V2,
                             dab.mesh_P)

    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    # debug(da_mod)
    # print(np.size(dab.mod_zvs_phi), '-----------------------------------(dab.mod_zvs_phi)', dab.mod_zvs_phi)
    # print(np.size(dab.mod_zvs_tau1),'-----------------------------------(dab.mod_zvs_tau1)', dab.mod_zvs_tau1)
    # print(np.size(dab.mod_zvs_tau2),'-----------------------------------(dab.mod_zvs_tau2)', dab.mod_zvs_tau2)

    # debug('phi min:', np.nanmin(dab.mod_zvs_phi), 'phi max:', np.nanmax(dab.mod_zvs_phi))
    zvs_coverage = np.count_nonzero(dab.mod_zvs_mask_zvs) / np.size(dab.mod_zvs_mask_zvs)
    # print(dab.mod_zvs_mask_zvs)
    debug('zvs coverage:', zvs_coverage)

    ## Plotting
    info("\nStart Plotting\n")
    import plot_dab

    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    directory = os.path.join(directory, 'results')
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = os.path.join(directory, 'zvs_mod')
    if not os.path.exists(directory):
        os.mkdir(directory)
    name = 'mod_zvs'
    comment = 'Only modulation calculation results for mod_zvs with V1 {}, V2 {} and P {} steps.'.format(
        int(dab.V1_step),
        int(dab.V2_step),
        int(dab.P_step))

    plt = plot_dab.Plot_DAB(latex=True)

    # Plot OptZVS mod results
    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab.mesh_P)[1] / 2)

    debug('View plane: U_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0]))
    # Plot all modulation angles
    # plt.new_fig(nrows=1, ncols=3,
    #             tab_title='OptZVS Modulation Angles (U_1 = {:.1f}V)'.format(dab.mesh_V1[0, v1_middle, 0]))
    plt.plot_modulation(dab.mesh_P[:, v1_middle, :],
                        dab.mesh_V2[:, v1_middle, :],
                        dab.mod_zvs_phi[:, v1_middle, :],
                        dab.mod_zvs_tau1[:, v1_middle, :],
                        dab.mod_zvs_tau2[:, v1_middle, :],
                        mask1=dab.mod_zvs_mask_Im2[:, v1_middle, :],
                        mask2=dab.mod_zvs_mask_IIm2[:, v1_middle, :],
                        mask3=dab.mod_zvs_mask_IIIm1[:, v1_middle, :],
                        maskZVS=dab.mod_zvs_mask_zvs[:, v1_middle, :],
                        tab_title='OptZVS Modulation Angles (U_1 = {:.1f}V)'.format(dab.mesh_V1[0, v1_middle, 0])
                        )
    fname = name + '_V1_{:.0f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    fcomment = comment + ' View plane: V_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    plt.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot all modulation angles but separately with autoscale
    plt.new_fig(nrows=1, ncols=3, tab_title='OptZVS Modulation Angles (autoscale)')
    plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                         dab.mesh_V2[:, v1_middle, :],
                         dab.mod_zvs_phi[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][0],
                         xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                         dab.mesh_V2[:, v1_middle, :],
                         dab.mod_zvs_tau1[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][1],
                         xlabel='P / W', ylabel='U2 / V', title='tau1 in rad')
    plt.subplot_contourf(dab.mesh_P[:, v1_middle, :],
                         dab.mesh_V2[:, v1_middle, :],
                         dab.mod_zvs_tau2[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][2],
                         xlabel='P / W', ylabel='U2 / V', title='tau2 in rad')
    fname = name + '_V1_{:.0f}V_autoscale'.format(dab.mesh_V1[0, v1_middle, 0])
    fcomment = comment + ' View plane: V_1 = {:.1f}V'.format(dab.mesh_V1[0, v1_middle, 0])
    plt.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot a cross-section through the V2 plane
    v2_middle = int(np.shape(dab.mesh_P)[0] / 2)
    debug('View plane: U_2 = {:.1f}V'.format(dab.mesh_V2[v2_middle, 0, 0]))
    # Plot all modulation angles
    # plt.new_fig(nrows=1, ncols=3,
    #             tab_title='OptZVS Modulation Angles (U_2 = {:.1f}V)'.format(dab.mesh_V2[v2_middle, 0, 0]))
    plt.plot_modulation(dab.mesh_P[v2_middle, :, :],
                        dab.mesh_V1[v2_middle, :, :],
                        dab.mod_zvs_phi[v2_middle, :, :],
                        dab.mod_zvs_tau1[v2_middle, :, :],
                        dab.mod_zvs_tau2[v2_middle, :, :],
                        mask1=dab.mod_zvs_mask_Im2[v2_middle, :, :],
                        mask2=dab.mod_zvs_mask_IIm2[v2_middle, :, :],
                        mask3=dab.mod_zvs_mask_IIIm1[v2_middle, :, :],
                        maskZVS=dab.mod_zvs_mask_zvs[v2_middle, :, :],
                        Vnum=1,
                        tab_title='OptZVS Modulation Angles (U_2 = {:.1f}V)'.format(dab.mesh_V2[v2_middle, 0, 0])
                        )
    fname = name + '_V2_{:.0f}V'.format(dab.mesh_V2[v2_middle, 0, 0])
    fcomment = comment + ' View plane: V_2 = {:.1f}V'.format(dab.mesh_V2[v2_middle, 0, 0])
    plt.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot Coss
    plt.new_fig(nrows=1, ncols=2, tab_title='Coss ' + mosfet1, sharex=False, sharey=False)
    plt.subplot(np.arange(dab['coss_' + mosfet1].shape[0]),
                dab['coss_' + mosfet1],
                ax=plt.figs_axes[-1][1][0],
                xlabel='U_DS / V', ylabel='C_oss / pF', title='Coss ' + mosfet1,
                yscale='log')
    plt.subplot(np.arange(dab['qoss_' + mosfet1].shape[0]),
                dab['qoss_' + mosfet1],
                ax=plt.figs_axes[-1][1][1],
                xlabel='U_DS / V', ylabel='Q_oss / nC', title='Qoss ' + mosfet1)

    plt.show()
