import numpy as np
import dab_datasets as ds
from debug_tools import *
import matplotlib.pyplot as plt
import optuna
import csv
import pandas as pd

import Irms_Calc

# The dict keys this modulation will return
MOD_KEYS = ['phi', 'tau1', 'tau2', 'mask_zvs', 'mask_Im2', 'mask_IIm2',
            'mask_IIIm1', 'zvs_coverage', 'zvs_coverage_notnan', 'I_rms_Mean', 'error']


#
# def I_integral_Cal_HF1(n, Ls, Lc1, fs: np.ndarray | int | float, mode,
#                        V1: np.ndarray, V2: np.ndarray,
#                        phi: np.ndarray, tau1: np.ndarray, tau2: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
#
#     I_integrated = np.zeros_like(phi)
#     i_alpha, i_beta, i_gamma, i_delta = (np.full_like(phi, np.nan) for _ in range(4))
#
#     ws = 2 * np.pi * fs
#     iL_factor = V1 / (ws * Ls)
#     iLc1_factor = V1 / (ws * Lc1)
#     d = n * V2 / V1
#
#     alpha = 3.14159 - tau1
#     beta = 3.14159 + phi - tau2
#     gamma = np.full_like(phi, 3.14159)
#     delta = 3.14159 + phi
#     alpha_ = 2 * 3.14159 - tau1
#     beta_ = 2 * 3.14159 + phi - tau2
#     gamma_ = np.full_like(phi, 0)
#     delta_ = phi
#     gamma__ = np.full_like(phi, 2 * 3.14159)
#     if mode == 1:
#         i_alpha = iL_factor * (d * (-tau1 + tau2 * 0.5 - phi + np.pi) - tau1 * 0.5) - iLc1_factor * tau1 * 0.5
#         i_beta = iL_factor * (d * tau2 * 0.5 + tau1 * 0.5 - tau2 + phi) + iLc1_factor * (tau1 * 0.5 - tau2 + phi)
#         i_gamma = iL_factor * (d * (-tau2 * 0.5 + phi) + tau1 * 0.5) + iLc1_factor * tau1 * 0.5
#         i_delta = iL_factor * (-d * tau2 * 0.5 - tau1 * 0.5 - phi + np.pi) + iLc1_factor * (-tau1 * 0.5 - phi + np.pi)
#         # print("alpha =", alpha)
#         # print("beta =", beta)
#         # print("gamma =", gamma)
#         # print("delta =", delta)
#         # print("alpha_ =", alpha_)
#         # print("beta_ =", beta_)
#         # print("gamma_ =", gamma_)
#         # print("delta_ =", delta_)
#         # print("gamma__ =", gamma__)
#
#     if mode == 2:
#         i_alpha = iL_factor * (d * tau2 * 0.5 - tau1 * 0.5) - iLc1_factor * tau1 * 0.5
#         i_beta = iL_factor * (d * tau2 * 0.5 + tau1 * 0.5 - tau2 + phi) + iLc1_factor * (tau1 * 0.5 - tau2 + phi)
#         i_gamma = iL_factor * (-d * tau2 * 0.5 + tau1 * 0.5) + iLc1_factor * tau1 * 0.5
#         i_delta = iL_factor * (-d * tau2 * 0.5 + tau1 + phi) + iLc1_factor * (tau1 * 0.5 + phi)
#
#     i_alpha_ = -i_alpha
#     i_beta_ = -i_beta
#     i_gamma_ = -i_gamma
#     i_delta_ = -i_delta
#
#     # Create arrays for angles and currents
#     angles = np.array([gamma_, alpha, delta_, beta, gamma, alpha_, delta, beta_, gamma__])
#     currents = np.array([i_gamma_, i_alpha, i_delta_, i_beta, i_gamma, i_alpha_, i_delta, i_beta_, i_gamma_])
#
#     # Sort the angles and currents according to the angle order
#     sorted_indices = np.argsort(angles, axis=0)
#     x = np.take_along_axis(angles, sorted_indices, axis=0)
#     y = np.take_along_axis(currents, sorted_indices, axis=0)
#
#     for i in range(8):
#         I_integrated += (x[i + 1] - x[i])*0.33334*(y[i]**2 + y[i+1]**2 + y[i + 1]*y[i])
#
#     I_rms = np.sqrt(I_integrated / (2 * np.pi))
#     return I_rms, x, y # angles, currents
#
#
# def I_integral_Cal_Lc2(n, Ls, Lc2, fs: np.ndarray | int | float, mode,
#                        V1: np.ndarray, V2: np.ndarray,
#                        phi: np.ndarray | int | float, tau1: np.ndarray | int | float, tau2: np.ndarray | int | float) \
#         -> [np.ndarray, np.ndarray, np.ndarray]:
#
#     I_integrated = np.zeros_like(phi)
#     i_alpha, i_beta, i_gamma, i_delta = (np.full_like(phi, np.nan) for _ in range(4))
#
#     ws = 2 * np.pi * fs
#     # Transform Lc2 to side 1
#     Lc2_ = Lc2 * n ** 2
#     # Transform V2 to side 1
#     V2_ = V2 * n
#     iLc2_factor = V2_ / (ws * Lc2_)
#
#     alpha = 3.14159 - tau1
#     beta = 3.14159 + phi - tau2
#     gamma = np.full_like(phi, 3.14159)
#     delta = 3.14159 + phi
#     alpha_ = 2 * 3.14159 - tau1
#     beta_ = 2 * 3.14159 + phi - tau2
#     gamma_ = np.full_like(phi, 0)
#     delta_ = phi
#     gamma__ = np.full_like(phi, 2 * 3.14159)
#
#     if mode == 1:
#         i_alpha = iLc2_factor * (tau1 - tau2 * 0.5 + phi - np.pi)
#         i_beta = - iLc2_factor * (tau2 * 0.5)
#         i_gamma = iLc2_factor * (tau2 * 0.5 - phi)
#         i_delta = iLc2_factor * tau2 * 0.5
#
#     if mode == 2:
#         i_alpha = - iLc2_factor * tau2 * 0.5
#         i_beta = - iLc2_factor * tau2 * 0.5
#         i_gamma = iLc2_factor * tau2 * 0.5
#         i_delta = iLc2_factor * tau2 * 0.5
#
#     i_alpha_ = -i_alpha
#     i_beta_ = -i_beta
#     i_gamma_ = -i_gamma
#     i_delta_ = -i_delta
#
#     # Create arrays for angles and currents
#     angles = np.array([gamma_, alpha, delta_, beta, gamma, alpha_, delta, beta_, gamma__])
#     currents = np.array([i_gamma_, i_alpha, i_delta_, i_beta, i_gamma, i_alpha_, i_delta, i_beta_, i_gamma_])
#
#     # Sort the angles and currents according to the angle order
#     sorted_indices = np.argsort(angles, axis=0)
#     x = np.take_along_axis(angles, sorted_indices, axis=0)
#     y = np.take_along_axis(currents, sorted_indices, axis=0)
#
#     for i in range(8):
#         I_integrated += (x[i + 1] - x[i])*0.33334*(y[i]**2 + y[i+1]**2 + y[i + 1]*y[i])
#
#     I_Lc2_rms = np.sqrt(I_integrated / (2 * np.pi))
#     return I_Lc2_rms, x, y
#
#
# def PI_sigma_transformation(n, L, lc1, lc2) -> [float, float]:
#
#     L_total = L + lc1 + lc2 * n ** 2
#     Ls = ((lc1 + lc2 * n ** 2) * L) / L_total
#     Lm = (lc1 * lc2 * n ** 2) / L_total
#     return Ls, Lm


def calc_modulation(n, Ls, Lc1, Lc2, fs: np.ndarray | int | float, Coss1: np.ndarray, Coss2: np.ndarray,
                    V1: np.ndarray, V2: np.ndarray, P: np.ndarray, C_Par_flag) -> dict:
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
    zvs = np.full_like(V1, False)
    _Im2_mask = np.full_like(V1, False)
    _IIm2_mask = np.full_like(V1, False)
    _IIIm1_mask = np.full_like(V1, False)

    # np.set_printoptions(threshold=20000)

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

    C_Par1 = 6e-12
    C_Par2 = 6e-12
    if C_Par_flag == 1:
        Coss1 = 1.2 * (Coss1 + C_Par1)
        Coss2 = 1.2 * (Coss2 + C_Par2)
    # Calculate required Q for each voltage
    # FIXME Check if factor 2 is right here!
    Q_AB_req1 = _integrate_Coss(Coss1 * 2, V1)
    Q_AB_req2 = _integrate_Coss(Coss2 * 2, V2)

    # debug(Coss1)

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
    phi_III, tau1_III, tau2_III = _calc_interval_III(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

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
    # Int. I (mode 2): use results
    phi[np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)] = phi_I[
        np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)]
    # debug('phi1', phi)
    # debug('phi_I', phi_I)
    tau1[np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)] = tau1_I[
        np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)]
    tau2[np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)] = tau2_I[
        np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)]
    _Im2_mask = np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)
    # debug('_Im2_mask', _Im2_mask)
    zvs[np.bitwise_and(_phi_I_leq_zero_mask, _tau1_I_leq_pi_mask)] = True

    # Int. II (mode 2):
    # if tau1 <= pi:
    _tau1_II_leq_pi_mask = np.less_equal(tau1_II, np.pi)
    # debug('_tau1_II_leq_pi_mask', _tau1_II_leq_pi_mask)
    # if tau1 > pi:
    _tau1_II_g_pi_mask = np.greater(tau1_II, np.pi)
    # debug('_tau1_II_g_pi_mask', _tau1_II_g_pi_mask)
    # Int. II (mode 2): use results
    phi[np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)] = phi_II[
        np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)]
    tau1[np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)] = tau1_II[
        np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)]
    tau2[np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)] = tau2_II[
        np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)]
    _IIm2_mask = np.bitwise_and(_tau1_II_leq_pi_mask, _phi_I_g_zero_mask)
    # debug('_IIm2_mask', _IIm2_mask)
    zvs[np.bitwise_and(_phi_I_g_zero_mask, _tau1_II_leq_pi_mask)] = True

    phi_only_mode2 = np.around(phi, 5)
    tau1_only_mode2 = np.around(tau1, 5)
    tau2_only_mode2 = np.around(tau2, 5)

    # IL_rms calculation
    I_L_rms_m2, x_L_m2, y_L_m2 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 0, 2, V1, V2,
                                                 phi_only_mode2, tau1_only_mode2, tau2_only_mode2)

    # ILc1_rms calculation
    I_Lc1_rms_m2, x_Lc1_m2, y_Lc1_m2 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 1, 2, V1, V2,
                                                       phi_only_mode2, tau1_only_mode2, tau2_only_mode2)

    # ILc2_rms calculation
    I_Lc2_rms_m2, x_Lc2_m2, y_Lc2_m2 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 2, 2, V1, V2,
                                                       phi_only_mode2, tau1_only_mode2, tau2_only_mode2)

    # Int. III (mode 1):
    # if tau2 <= pi:
    _tau2_III_leq_pi_mask = np.less_equal(tau2_III, np.pi)
    # debug('_tau2_III_leq_pi_mask', _tau2_III_leq_pi_mask)
    # Int. III (mode 1): use results
    phi[np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)] = phi_III[
        np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)]
    # debug('phi3', phi)
    # debug('phi_III', phi_III)
    tau1[np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)] = tau1_III[
        np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)]
    tau2[np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)] = tau2_III[
        np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)]
    _IIIm1_mask = np.bitwise_and(_tau2_III_leq_pi_mask, _tau1_II_g_pi_mask)
    # debug('_IIIm1_mask', _IIIm1_mask)
    zvs[np.bitwise_and(_tau1_II_g_pi_mask, _tau2_III_leq_pi_mask)] = True

    # Int. I (mode 2): ZVS is analytically IMPOSSIBLE!
    zvs[np.bitwise_and(_phi_I_leq_zero_mask, np.bitwise_not(_tau1_I_leq_pi_mask))] = False
    # debug('zvs', zvs)
    # zvs = np.bitwise_not(np.bitwise_and(_phi_leq_zero_mask, np.bitwise_not(_tau1_leq_pi_mask)))
    # debug('zvs bitwise not', zvs)

    ## Recalculate phi for negative power
    phi_nP = - (tau1 + phi - tau2)
    phi[_negative_power_mask] = phi_nP[_negative_power_mask]

    phi = np.around(phi, 5)
    tau1 = np.around(tau1, 5)
    tau2 = np.around(tau2, 5)

    phi_only_mode1 = np.nan_to_num(phi, nan=0) - np.nan_to_num(phi_only_mode2, nan=0)
    tau1_only_mode1 = np.nan_to_num(tau1, nan=0) - np.nan_to_num(tau1_only_mode2, nan=0)
    tau2_only_mode1 = np.nan_to_num(tau2, nan=0) - np.nan_to_num(tau2_only_mode2, nan=0)
    phi_only_mode1 = np.where(phi_only_mode1 == 0, np.nan, phi_only_mode1)
    tau1_only_mode1 = np.where(tau1_only_mode1 == 0, np.nan, tau1_only_mode1)
    tau2_only_mode1 = np.where(tau2_only_mode1 == 0, np.nan, tau2_only_mode1)

    # IL_rms calculation
    I_L_rms_m1, x_L_m1, y_L_m1 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 0, 1, V1, V2,
                                                 phi_only_mode1, tau1_only_mode1, tau2_only_mode1)

    # ILc1_rms calculation
    I_Lc1_rms_m1, x_Lc1_m1, y_Lc1_m1 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 1, 1, V1, V2,
                                                       phi_only_mode1, tau1_only_mode1, tau2_only_mode1)

    # ILc2_rms calculation
    I_Lc2_rms_m1, x_Lc2_m1, y_Lc2_m1 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 2, 1, V1, V2,
                                                       phi_only_mode1, tau1_only_mode1, tau2_only_mode1)

    Irms_m1_mean = np.nanmean(np.nan_to_num(I_L_rms_m1, nan=0) +
                              np.nan_to_num(I_Lc1_rms_m1, nan=0) +
                              np.nan_to_num(I_Lc2_rms_m1, nan=0))

    Irms_m2_mean = np.nanmean(np.nan_to_num(I_L_rms_m2, nan=0) +
                              np.nan_to_num(I_Lc1_rms_m2, nan=0) +
                              np.nan_to_num(I_Lc2_rms_m2, nan=0))

    Irms = np.nanmean(Irms_m1_mean + Irms_m2_mean)

    # # mode 1
    # if tau2_only_mode1 >= phi_only_mode1 >= (np.pi - tau1_only_mode1):
    #     mode = 1
    # # mode 2
    # if 0 >= phi_only_mode2 >= (tau2_only_mode2 - tau1_only_mode2):
    #     mode = 2
    # Ilstart = np.nan_to_num(y_L_m1, 0) + np.nan_to_num(y_L_m2, 0)
    #
    # if (not np.isnan(V1[0][0][0]) and not np.isnan(V2[0][0][0]) and not np.isnan(phi[0][0][0]) and not
    # np.isnan(tau1[0][0][0]) and not np.isnan(tau2[0][0][0]) and not np.isnan(Ilstart[0][0][0][0])):
    #     Irms_Gecko = Irms_Calc.Irms_validation_Gecko(V1[0][0][0], V2[0][0][0], n, Ls, Lc1, Lc2,
    #                                                  phi[0][0][0], tau1[0][0][0], tau2[0][0][0], Ilstart[0][0][0][0])
    # else:
    #     Irms_Gecko = 0
    # # List of values
    # values = [I_L_rms_m1, I_Lc1_rms_m1, I_Lc2_rms_m1, I_L_rms_m2, I_Lc1_rms_m2, I_Lc2_rms_m2]
    #
    # # Calculate the sum excluding NaN values
    # Irms_Calculated = sum(value for value in values if not np.isnan(value))
    # # Calculate the relative error
    # relative_error = abs(Irms_Gecko - Irms_Calculated[0][0][0]) / Irms_Gecko
    # # Convert the relative error to percentage
    # error_percentage = relative_error * 100
    #
    # print('==========================================================')
    # print(f'error: {error_percentage:.2f} %')
    #
    # if mode == 1:
    #     print('mode:', mode)
    #     print('==========================================================')
    #     plot_Irms(x_L_m1, y_L_m1, x_Lc1_m1, y_Lc1_m1, x_Lc2_m1, y_Lc2_m1, V1[0][0][0], V2[0][0][0], P[0][0][0])
    # else:
    #     print('mode:', mode)
    #     print('==========================================================')
    #     plot_Irms(x_L_m2, y_L_m2, x_Lc1_m2, y_Lc1_m2, x_Lc2_m2, y_Lc2_m2, V1[0][0][0], V2[0][0][0], P[0][0][0])

    # Init return dict

    da_mod_results = dict()
    # Save the results in the dict
    # da_mod_results[MOD_KEYS[0]] = phi
    # Convert phi because the math from the paper uses Middle-Pulse alignment but we use First-Falling-Edge alignment!
    da_mod_results[MOD_KEYS[0]] = phi
    da_mod_results[MOD_KEYS[1]] = tau1
    da_mod_results[MOD_KEYS[2]] = tau2
    da_mod_results[MOD_KEYS[3]] = zvs
    da_mod_results[MOD_KEYS[4]] = _Im2_mask
    da_mod_results[MOD_KEYS[5]] = _IIm2_mask
    da_mod_results[MOD_KEYS[6]] = _IIIm1_mask

    ## ZVS coverage based on calculation
    da_mod_results[MOD_KEYS[7]] = np.count_nonzero(zvs) / np.size(zvs)
    ## ZVS coverage based on calculation
    da_mod_results[MOD_KEYS[8]] = np.count_nonzero(zvs[~np.isnan(tau1)]) / np.size(zvs[~np.isnan(tau1)])
    ## Irms
    da_mod_results[MOD_KEYS[9]] = Irms  # Irms_Calculated
    da_mod_results[MOD_KEYS[10]] = 0  #error_percentage
    # debug(da_mod_results)
    return da_mod_results


def plot_Irms(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, V1, V2,
              P):
    # Load the CSV files
    file1 = "C:/Users/vijay/Desktop/UPB/DAB_MOSFET_Modulation_v3_I_HF1.csv"
    df = pd.read_csv(file1)
    X_shift_value = df['# t'][0]
    df['t_shifted'] = df['# t'] - X_shift_value

    x0_ = x0 + np.full_like(x0, 3.1415 * 2)
    x0 = 2 * 3.9788e-7 * np.append(x0, x0_)
    y0 = np.append(y0, y0)

    x1_ = x1 + np.full_like(x1, 3.1415 * 2)
    x1 = 2 * 3.9788e-7 * np.append(x1, x1_)
    y1 = np.append(y1, y1)

    x2_ = x2 + np.full_like(x2, 3.1415 * 2)
    x2 = 2 * 3.9788e-7 * np.append(x2, x2_)
    y2 = np.append(y2, y2)

    # Create a new folder to save the image
    output_folder_ = '../results/currents'
    os.makedirs(output_folder_, exist_ok=True)

    # Define the file name for the image
    image_name_ = f'Irms_Calc_vs_Gecko_{V1:.0f}_{V2:.0f}_{P:.0f}_plot.png'
    image_path_ = os.path.join(output_folder_, image_name_)
    # Create the plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(x0, y0, 'o-r', ms=3, label='Calc.')
    axs[0].plot(df['t_shifted'], df['i_Ls'], color='b', label='Gecko')
    axs[0].set_title('Plot of i_L')
    axs[0].grid(color='gray', linestyle='--', linewidth=0.5)

    axs[1].plot(x1, y1, 'o-r', ms=3, label='Calc.')
    axs[1].plot(df['t_shifted'], df['i_Lc1'], color='b', label='Gecko')
    axs[1].set_title('Plot of i_Lc1')
    axs[1].grid(color='gray', linestyle='--', linewidth=0.5)

    axs[2].plot(x2, y2, 'o-r', ms=3, label='Calc.')
    axs[2].plot(df['t_shifted'], df['i_Lc2_'], color='b', label='Gecko')
    axs[2].set_title('Plot of i_Lc2')
    axs[2].grid(color='gray', linestyle='--', linewidth=0.5)

    # Add a common xlabel and ylabel
    fig.text(0.5, 0.04, 'Time (Sec)', ha='center', va='center')
    fig.text(0.04, 0.5, 'Current (A)', ha='center', va='center', rotation='vertical')

    # Add a common legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.grid(True)
    ## Maximize the window to fullscreen
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.show()

    plt.savefig(image_path_, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {image_path_}")

    # Shift the t values
    # Load the CSV file
    # file1 = "C:/Users/vijay/Desktop/UPB/DAB_MOSFET_Modulation_v3_I_HF1.csv"
    # df = pd.read_csv(file1)
    #
    # X_shift_value = df['# t'][0]
    # df['t_shifted'] = df['# t'] - X_shift_value
    #
    # # Shifts for i_Ls, i_Lc1, i_Lc2
    # Y_shift_iL = df['i_Ls'][0] - y0[0][0][0][0]
    # df['i_Ls_shifted'] = df['i_Ls'] + Y_shift_iL
    #
    # Y_shift_iLc1 = df['i_Lc1'][0] - y1[0][0][0][0]
    # df['i_Lc1_shifted'] = df['i_Lc1'] + Y_shift_iLc1
    #
    # Y_shift_iLc2 = df['i_Lc2_'][0] - y2[0][0][0][0]
    # df['i_Lc2_shifted'] = df['i_Lc2_'] + Y_shift_iLc2
    #
    # # Duplicate and shift the minimal data points
    # x0_ = x0 + np.full_like(x0, 3.1415 * 2)
    # x0 = 2 * 3.9788e-7 * np.append(x0, x0_)
    # # x0_ = x0 + np.full_like(x0, (1/200000))
    # # x0 = np.append(x0, x0_)
    # y0 = np.append(y0, y0)
    #
    # x1_ = x1 + np.full_like(x1, 3.1415 * 2)
    # x1 = 2 * 3.9788e-7 * np.append(x1, x1_)
    # # x1_ = x1 + np.full_like(x1, (1/200000))
    # # x1 = np.append(x1, x1_)
    # y1 = np.append(y1, y1)
    #
    # x2_ = x2 + np.full_like(x2, 3.1415 * 2)
    # x2 = 2 * 3.9788e-7 * np.append(x2, x2_)
    # # x2_ = x2 + np.full_like(x2, (1/200000))
    # # x2 = np.append(x2, x2_)
    # y2 = np.append(y2, y2)
    #
    # # Filter the dense data to match x-coordinates of the minimal data points
    # def filter_dense_data(x_minimal, x_dense, y_dense):
    #     indices = np.searchsorted(x_dense, x_minimal)
    #     indices = np.clip(indices, 0, len(x_dense) - 1)  # Ensure indices are within valid range
    #     x_filtered = x_dense[indices]
    #     y_filtered = y_dense[indices]
    #     return x_filtered, y_filtered
    #
    # # Apply the filtering
    # x_dense_filtered_0, y_dense_filtered_0 = filter_dense_data(x0, df['t_shifted'], df['i_Ls_shifted'])
    # x_dense_filtered_1, y_dense_filtered_1 = filter_dense_data(x1, df['t_shifted'], df['i_Lc1_shifted'])
    # x_dense_filtered_2, y_dense_filtered_2 = filter_dense_data(x2, df['t_shifted'], df['i_Lc2_shifted'])
    #
    # # Convert x_dense_filtered_0 to numpy array if it's a Series
    # x_dense_filtered_00 = x_dense_filtered_0.to_numpy() if isinstance(x_dense_filtered_0,
    #                                                                  pd.Series) else x_dense_filtered_0
    #
    # # Convert y_dense_filtered_0 to numpy array if it's a Series
    # y_dense_filtered_00 = y_dense_filtered_0.to_numpy() if isinstance(y_dense_filtered_0,
    #                                                                  pd.Series) else y_dense_filtered_0
    #
    # # Convert x_dense_filtered_1 to numpy array if it's a Series
    # x_dense_filtered_10 = x_dense_filtered_1.to_numpy() if isinstance(x_dense_filtered_1,
    #                                                                  pd.Series) else x_dense_filtered_1
    #
    # # Convert y_dense_filtered_1 to numpy array if it's a Series
    # y_dense_filtered_10 = y_dense_filtered_1.to_numpy() if isinstance(y_dense_filtered_1,
    #                                                                  pd.Series) else y_dense_filtered_1
    #
    # # Convert x_dense_filtered_2 to numpy array if it's a Series
    # x_dense_filtered_20 = x_dense_filtered_2.to_numpy() if isinstance(x_dense_filtered_2,
    #                                                                  pd.Series) else x_dense_filtered_2
    #
    # # Convert y_dense_filtered_2 to numpy array if it's a Series
    # y_dense_filtered_20 = y_dense_filtered_2.to_numpy() if isinstance(y_dense_filtered_2,
    #                                                                  pd.Series) else y_dense_filtered_2
    #
    # # RMS calculation function
    # def calculate_rms(x, y):
    #     I_integrated = 0
    #     for i in range(len(x) - 1):
    #         I_integrated += (x[i + 1] - x[i]) * 0.33334 * (y[i] ** 2 + y[i + 1] ** 2 + y[i + 1] * y[i])
    #     I_rms = np.sqrt(I_integrated * 1e5)
    #     return I_rms
    #
    # Irmscal = calculate_rms(x0, y0) + calculate_rms(x1, y1) + calculate_rms(x2, y2)
    # IrmsGecko = (calculate_rms(x_dense_filtered_00, y_dense_filtered_00) +
    #              calculate_rms(x_dense_filtered_10, y_dense_filtered_10) +
    #              calculate_rms(x_dense_filtered_20, y_dense_filtered_20))
    #
    # # Calculate the relative error
    # relative_error = (IrmsGecko - Irmscal) / IrmsGecko
    # # Convert the relative error to percentage
    # error_percentage = relative_error * 100
    #
    # print(f'shifted__error: {error_percentage:.2f} %')
    # print('==========================================================')
    #
    # # Plotting the data
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.plot(x0, y0, 'o-r', ms=3, label='Calc.')
    # plt.plot(x_dense_filtered_0, y_dense_filtered_0, 'o-b', ms=3, label='Gecko')
    # plt.plot(df['t_shifted'], df['i_Ls'], '-g', label='CSV')
    # plt.title('Plot of i_L')
    # plt.xlabel('Time (Sec)')
    # plt.ylabel('iL (A)')
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # plt.legend()
    #
    # plt.subplot(1, 3, 2)
    # plt.plot(x1, y1, 'o-r', ms=3, label='Calc.')
    # plt.plot(x_dense_filtered_1, y_dense_filtered_1, 'o-b', ms=3, label='Gecko')
    # plt.plot(df['t_shifted'], df['i_Lc1'], '-g', label='CSV')
    # plt.title('Plot of i_Lc1')
    # plt.xlabel('Time (Sec)')
    # plt.ylabel('iLc1 (A)')
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # plt.legend()
    #
    # plt.subplot(1, 3, 3)
    # plt.plot(x2, y2, 'o-r', ms=3, label='Calc.')
    # plt.plot(x_dense_filtered_2, y_dense_filtered_2, 'o-b', ms=3, label='Gecko')
    # plt.plot(df['t_shifted'], df['i_Lc2_'], '-g', label='CSV')
    # plt.title('Plot of i_Lc2')
    # plt.xlabel('Time (Sec)')
    # plt.ylabel('iLc2 (A)')
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # plt.legend()
    #
    # # Maximize the window to fullscreen
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.show()


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


def objective(trial):
    # Set the basic DAB Specification
    dab = ds.DAB_Data()
    dab.V1_nom = 700
    dab.V1_min = 600
    dab.V1_max = 800
    dab.V1_step = 3
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = 25 * 3
    # dab.V2_step = 4
    dab.P_min = -2200
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 19 * 3
    dab.fs = 200000
    # Generate meshes
    dab.gen_meshes()
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
    dab.Ls = trial.suggest_float("dab.Ls", 80e-6, 90e-6)
    dab.n = trial.suggest_float("dab.n", 4.2, 4.2)
    dab.Lc1 = trial.suggest_float("dab.Lc1", 0.1 * dab.Ls, 100 * dab.Ls)
    dab.Lc2 = trial.suggest_float("dab.Lc2", 0.1 * dab.Ls, 100 * dab.Ls)
    C_Par_flag = 1
    da_mod = calc_modulation(dab.n,
                             dab.Ls,
                             dab.Lc1,
                             dab.Lc2,
                             dab.fs,
                             dab['coss_' + mosfet1],
                             dab['coss_' + mosfet2],
                             dab.mesh_V1,
                             dab.mesh_V2,
                             dab.mesh_P,
                             C_Par_flag)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    zvs_coverage = np.count_nonzero(dab.mod_zvs_mask_zvs) / np.size(dab.mod_zvs_mask_zvs)
    I_Mean = dab.mod_zvs_I_rms_Mean
    # debug('Mean', Mean)
    # return zvs_coverage, Mean

    # Introduce the factor
    factored_zvs_coverage = 1 * zvs_coverage
    factored_I_Mean = 1 * I_Mean

    # Store the original zvs_coverage as a user attribute
    trial.set_user_attr("original_zvs_coverage", zvs_coverage)
    trial.set_user_attr("original_Mean", I_Mean)

    return factored_zvs_coverage, factored_I_Mean


def find_optimal_zvs_coverage(): # (x, y, z) -> float:
    # Set the basic DAB Specification
    dab = ds.DAB_Data()
    dab.V1_nom = 700
    dab.V1_min = 600
    dab.V1_max = 800
    dab.V1_step = 3
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = 25 * 3
    # dab.V2_step = 4
    dab.P_min = -2200
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 19 * 3
    dab.fs = 200000
    dab.Lm = 595e-6
    C_Par_flag = 1

    dab.Ls = 85e-6
    dab.n = 4.2
    dab.Lc1 = 3 * 85e-6
    dab.Lc2 = 3 * 85e-6

    # Generate meshes
    dab.gen_meshes()
    # debug('mesh_V1', np.size(dab.mesh_V1))

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

    # for dab.Lc1 in np.arange(0.000356 * 0.5, 0.000356 * 1.5, 10e-6):  # 3e-4, 5e-4, 10e-6
    #     for dab.Lc2 in np.arange(6.9e-05 * 0.5, 6.9e-05 * 1.5, 5e-6):  # 6e-4, 35e-4, 25e-6
    da_mod = calc_modulation(dab.n,
                             dab.Ls,
                             dab.Lc1,
                             dab.Lc2,
                             dab.fs,
                             dab['coss_' + mosfet1],
                             dab['coss_' + mosfet2],
                             dab.mesh_V1,
                             dab.mesh_V2,
                             dab.mesh_P,
                             C_Par_flag)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    # debug(dab.mod_zvs_phi)
    zvs_coverage = np.count_nonzero(dab.mod_zvs_mask_zvs) / np.size(dab.mod_zvs_mask_zvs)
    print(f'n: {dab.n}, Lc1: {dab.Lc1}, Lc2: {dab.Lc2}, zvs_coverage: {zvs_coverage}')
    Mean = dab.mod_zvs_I_rms_Mean
    print(f'I_Mean: {Mean} A')
    # return dab.mod_zvs_error


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start.........")

    # find_optimal_zvs_coverage()
    # relative_error_percentage = find_optimal_zvs_coverage(700, 235, 200)

    # # Fixed value for v1
    # v1 = 700
    #
    # # Define specific values for v2
    # v2_values = [175, 235, 275]
    #
    # # Define corresponding colors for each v2 value
    # colors = ['red', 'blue', 'green']
    #
    # # List to store data for plotting
    # all_errors = []
    # all_powers = []
    #
    # # List to store average errors
    # average_errors = []
    #
    # # List to store all errors to calculate the overall average
    # total_errors = []
    #
    # # Iterate over different values of v2
    # for v2 in v2_values:
    #     errors = []
    #     powers = []
    #
    #     # Iterate over different power values
    #     for p in range(200, 2201, 500):
    #         # Calculate error for the current combination of v1, v2, and p
    #         error = find_optimal_zvs_coverage(v1, v2, p)
    #         errors.append(error)
    #         powers.append(p)
    #
    #     # Append errors and powers to the lists
    #     all_errors.append(errors)
    #     all_powers.append(powers)
    #
    #     # Calculate and store the average error for the current v2
    #     avg_error = sum(errors) / len(errors)
    #     average_errors.append(avg_error)
    #
    #     # Append the errors to total_errors to calculate overall average later
    #     total_errors.extend(errors)
    #
    # # Calculate the overall average error
    # overall_avg_error = sum(total_errors) / len(total_errors)
    #
    # # Plotting all graphs in a single plot
    # plt.figure(figsize=(10, 8))  # Adjust figure size if needed
    # # Plot each combination of v2 with specified colors
    # for i in range(len(v2_values)):
    #     plt.plot(all_powers[i], all_errors[i], marker='o', linestyle='-', color=colors[i], label=f'v2={v2_values[i]}')
    #
    # plt.title(f'Error Rate vs Power for v1={v1}')
    # plt.xlabel('Power (W)')
    # plt.ylabel('Error (%)')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # # Print the average errors for each v2 value
    # for i, v2 in enumerate(v2_values):
    #     print(f'Average error for v2={v2}: {average_errors[i]:.2f}%')
    #
    # # Print the overall average error
    # print(f'Overall average error: {overall_avg_error:.2f}%')

    Trails = 50
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=Trails)

    # Filter the trials based on the given criteria
    zvs_threshold = 0.97
    I_rms_threshold = 4
    filtered_trials = [
        trial for trial in study.trials
        if trial.values and trial.values[0] > zvs_threshold and trial.values[1] < I_rms_threshold
    ]

    # Check if there are any filtered trials
    if not filtered_trials:
        print("------------- No valid trials found. Please change the trial numbers or filter parameters -------------")
    # Extract data from the filtered trials
    filtered_data = [
        {
            'trial_number': trial.number,
            'zvs_coverage': trial.user_attrs["original_zvs_coverage"],
            'I_rms_Mean': trial.user_attrs["original_Mean"],
            'params': trial.params
        }
        for trial in filtered_trials
    ]

    # To display the original Pareto front
    # fig = optuna.visualization.plot_pareto_front(study, target_names=["zvs_coverage", "I_rms_Mean"])
    fig = optuna.visualization.plot_pareto_front(
        study,
        target_names=["zvs_coverage", "I_rms_Mean"],
        targets=lambda t: (t.user_attrs["original_zvs_coverage"], t.user_attrs["original_Mean"])
    )
    # Add annotations for zvs_threshold, I_rms_threshold, and number of trials
    fig.add_annotation(
        text=f"zvs_threshold: {zvs_threshold}<br>I_rms_threshold: {I_rms_threshold}<br>n_trials: {len(study.trials)}",
        xref="paper", yref="paper",
        x=0.5, y=1, showarrow=False,
        xanchor='center', yanchor='top',
        font=dict(size=12)
    )
    fig.show()

    # Displaying the filtered trial data
    for data in filtered_data:
        print(
            f"Trial Number: {data['trial_number']}, zvs_coverage: {data['zvs_coverage']}, I_rms_Mean: {data['I_rms_Mean']}, Params: {data['params']}")

    # Create a unique filename based on filter values and number of trials
    filename = f"pareto_front_zvs_{zvs_threshold}_I_rms_{I_rms_threshold}A_trials_{len(study.trials)}.html"
    # Specify the directory to save the file
    save_dir = r"C:\Users\vijay\Desktop\UPB\Thesis\dab_optimizer\results\optuna"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    # Combine directory and filename
    file_path = os.path.join(save_dir, filename)
    # Save the original Pareto front plot as an HTML file
    fig.write_html(file_path)
    print('file_path:', file_path)

    # # Save filtered data to CSV file
    # csv_filename = f"filtered_data_zvs_{zvs_threshold}_I_rms_{I_rms_threshold}A_trials_{len(study.trials)}.csv"
    # # Specify the directory to save the file
    # save_dir = r"C:\Users\vijay\Desktop\UPB\Thesis\dab_optimizer\results\optuna"
    # os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    # # Combine directory and filename
    # csv_file_path = os.path.join(save_dir, csv_filename)
    # # Write filtered data to CSV file
    # with open(csv_file_path, mode='w', newline='') as csv_file:
    #     fieldnames = ['trial_number', 'zvs_coverage', 'I_rms_Mean', 'params']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for data in filtered_data:
    #         writer.writerow(data)
    # print(f"Filtered data saved to: {csv_file_path}")
