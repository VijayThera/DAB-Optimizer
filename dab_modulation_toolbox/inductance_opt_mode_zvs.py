import numpy as np
import dab_datasets as ds
from debug_tools import *

import optuna
from datetime import datetime
import matplotlib.pyplot as plt
import Irms_Calc
import mod_zvs

# The dict keys this modulation will return
MOD_KEYS = ['phi', 'tau1', 'tau2', 'mask_zvs', 'mask_Im2', 'mask_IIm2',
            'mask_IIIm1', 'zvs_coverage', 'zvs_coverage_notnan', 'I_rms_cost', 'error']


def calc_modulation(n, Ls, Lc1, Lc2, fs: np.ndarray | int | float, Coss1: np.ndarray, Coss2: np.ndarray,
                    V1: np.ndarray, V2: np.ndarray, P: np.ndarray, Gecko_validation: bool) -> dict:
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
    # parasitic capacitance with copper blocks, TIM, and heatsink
    C_Par1 = 42e-12 # 6e-12
    C_Par2 = 42e-12 # 6e-12
    # 20% higher for safety margin
    C_total_1 = 1.2 * (Coss1 + C_Par1)
    C_total_2 = 1.2 * (Coss2 + C_Par2)
    # Calculate required Q for each voltage
    Q_AB_req1 = mod_zvs._integrate_Coss(C_total_1 * 2, V1)
    Q_AB_req2 = mod_zvs._integrate_Coss(C_total_2 * 2, V2)

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
    phi_I, tau1_I, tau2_I = mod_zvs._calc_interval_I(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

    # Int. II (mode 2): calculate phi, tau1 and tau2
    phi_II, tau1_II, tau2_II = mod_zvs._calc_interval_II(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

    # Int. III (mode 1): calculate phi, tau1 and tau2
    phi_III, tau1_III, tau2_III = mod_zvs._calc_interval_III(n, Ls, Lc1, Lc2_, ws, Q_AB_req1, Q_AB_req2, V1, V2_, I1)

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

    phi_m2 = np.around(phi, 5)
    tau1_m2 = np.around(tau1, 5)
    tau2_m2 = np.around(tau2, 5)

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

    phi_m1 = np.nan_to_num(phi, nan=0) - np.nan_to_num(phi_m2, nan=0)
    tau1_m1 = np.nan_to_num(tau1, nan=0) - np.nan_to_num(tau1_m2, nan=0)
    tau2_m1 = np.nan_to_num(tau2, nan=0) - np.nan_to_num(tau2_m2, nan=0)
    phi_m1 = np.where(phi_m1 == 0, np.nan, phi_m1)
    tau1_m1 = np.where(tau1_m1 == 0, np.nan, tau1_m1)
    tau2_m1 = np.where(tau2_m1 == 0, np.nan, tau2_m1)

    # # Masks for positive and negative powers
    # _positive_power_mask = np.greater(phi, 0)
    # _negative_power_mask = np.less_equal(phi, 0)

    # # Mode 1 positive and negative powers
    # phi_m1_p = np.where(_positive_power_mask, phi_m1, 0)
    # tau1_m1_p = np.where(_positive_power_mask, tau1_m1, 0)
    # tau2_m1_p = np.where(_positive_power_mask, tau2_m1, 0)
    #
    # phi_m1_n = np.where(_negative_power_mask, phi_m1, 0)
    # tau1_m1_n = np.where(_negative_power_mask, tau1_m1, 0)
    # tau2_m1_n = np.where(_negative_power_mask, tau2_m1, 0)

    # # Mode 2 positive and negative powers
    # phi_m2_p = np.where(_positive_power_mask, phi_m2, 0)
    # tau1_m2_p = np.where(_positive_power_mask, tau1_m2, 0)
    # tau2_m2_p = np.where(_positive_power_mask, tau2_m2, 0)
    #
    # phi_m2_n = np.where(_negative_power_mask, phi_m2, 0)
    # tau1_m2_n = np.where(_negative_power_mask, tau1_m2, 0)
    # tau2_m2_n = np.where(_negative_power_mask, tau2_m2, 0)

    # Ensure phi < 0 for mode1 and mode2 (negative power only)
    # phi_m1_p = np.where(phi_m1_p < 0, phi_m1_p, 0)
    # phi_m1_n = np.where(phi_m1_n < 0, phi_m1_n, 0)
    # phi_m2_p = np.where(phi_m2_p < 0, phi_m2_p, 0)
    # phi_m2_n = np.where(phi_m2_n < 0, phi_m2_n, 0)

    # Print Mode 1 positive and negative powers
    # print("Mode 1 Positive Powers:")
    # print("phi_m1_p:", phi_m1_p)
    # print("tau1_m1_p:", tau1_m1_p)
    # print("tau2_m1_p:", tau2_m1_p)

    # print("\nMode 1 Negative Powers:")
    # print("phi_m1_n:", phi_m1_n)
    # print("tau1_m1_n:", tau1_m1_n)
    # print("tau2_m1_n:", tau2_m1_n)

    # Print Mode 2 positive and negative powers
    # print("\nMode 2 Positive Powers:")
    # print("phi_m2_p:", phi_m2_p)
    # print("tau1_m2_p:", tau1_m2_p)
    # print("tau2_m2_p:", tau2_m2_p)

    # print("\nMode 2 Negative Powers:")
    # print("phi_m2_n:", phi_m2_n)
    # print("tau1_m2_n:", tau1_m2_n)
    # print("tau2_m2_n:", tau2_m2_n)

    # Irms_m1_mean = np.nanmean(np.nan_to_num(I_L_rms_m1, nan=0) +
    #                           np.nan_to_num(I_Lc1_rms_m1, nan=0) +
    #                           np.nan_to_num(I_Lc2_rms_m1, nan=0))

    # Irms_m2_mean = np.nanmean(np.nan_to_num(I_L_rms_m2, nan=0) +
    #                           np.nan_to_num(I_Lc1_rms_m2, nan=0) +
    #                           np.nan_to_num(I_Lc2_rms_m2, nan=0))

    # I_HF1 = np.nanmean(np.nan_to_num(I_L_rms_m1, nan=0) +
    #                    np.nan_to_num(I_Lc1_rms_m1, nan=0) +
    #                    np.nan_to_num(I_L_rms_m2, nan=0) +
    #                    np.nan_to_num(I_Lc1_rms_m2, nan=0))

    # I_HF2 = n * np.nanmean((np.nan_to_num(I_L_rms_m1, nan=0) +
    #                         np.nan_to_num(I_L_rms_m2, nan=0)) -
    #                        (np.nan_to_num(I_Lc2_rms_m1, nan=0) +
    #                         np.nan_to_num(I_Lc2_rms_m1, nan=0)))

    # Irms = np.nanmean(Irms_m1_mean + Irms_m2_mean)

    # debug('phi', phi, 'tau1', tau1, 'tau2', tau2)

    # I_cost = I_HF1 ** 2 + I_HF2 ** 2
    # print(f'I_HF1:{I_HF1}A, I_HF2:{I_HF2}A')
    # print(f'I_cost:{I_cost}')

    # Irms_Calc.plot_Irms(x_L_m1, y_L_m1, x_Lc1_m1, y_Lc1_m1, x_Lc2_m1, y_Lc2_m1, V1[0][0][0], V2[0][0][0], P[0][0][0])

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

    # if P < 0:
    #     mode = -1
    #     # print(f'mode:{mode}, P:{P}')
    # # mode = 1
    # # mode 1
    # debug('Ls', Ls, 'Lc1', Lc1, 'Lc2', Lc2, V1, V2)

    # # IL_rms calculation
    # I_L_rms_m1n, x_L_m1n, y_L_m1n = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 0, -1, V1, V2,
    #                                                 phi_m1_n, tau1_m1_n, tau2_m1_n)
    #
    # # ILc1_rms calculation
    # I_Lc1_rms_m1n, x_Lc1_m1n, y_Lc1_m1n = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 1, -1, V1, V2,
    #                                                       phi_m1_n, tau1_m1_n, tau2_m1_n)
    #
    # # ILc2_rms calculation
    # I_Lc2_rms_m1n, x_Lc2_m1n, y_Lc2_m1n = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 2, -1, V1, V2,
    #                                                       phi_m1_n, tau1_m1_n, tau2_m1_n)

    # Ilstart = np.nan_to_num(y_L_m1p, 0) + np.nan_to_num(y_L_m2, 0) #+ np.nan_to_num(y_L_m1n, 0)
    # Irms_Calc.Irms_validation_Gecko(V1[0][0][0], V2[0][0][0], n, Ls, Lc1, Lc2, phi[0][0][0], tau1[0][0][0], tau2[0][0][0], Ilstart[0][0][0][0])

    # if mode == -1:
    #     print('mode:', mode)
    #     Irms_Calc.plot_Irms(x_L_m1n, y_L_m1n, x_Lc1_m1n, y_Lc1_m1n, x_Lc2_m1n, y_Lc2_m1n, V1[0][0][0], V2[0][0][0],
    #                         P[0][0][0], mode)
    # if mode == 1:
    #     print('mode:', mode)
    #     Irms_Calc.plot_Irms(x_L_m1p, y_L_m1p, x_Lc1_m1p, y_Lc1_m1p, x_Lc2_m1p, y_Lc2_m1p, V1[0][0][0], V2[0][0][0],
    #                         P[0][0][0], mode)
    # if mode == 2:
    #     print('mode:', mode)
    #     Irms_Calc.plot_Irms(x_L_m2, y_L_m2, x_Lc1_m2, y_Lc1_m2, x_Lc2_m2, y_Lc2_m2, V1[0][0][0], V2[0][0][0],
    #                         P[0][0][0], mode)
    #===================================================
    # if()
    # I_cost = Irms_Calc.I_cost(n, Ls, Lc1, Lc2, fs, V1, V2, phi_m1, tau1_m1, tau2_m1, phi_m2, tau1_m2, tau2_m2)
    # print(f'I_cost:{I_cost}')
    if Gecko_validation:
        if tau2 >= phi >= (np.pi - tau1):
            mode = 1
            # debug('phi_m1:', phi_m1, 'tau1_m1:', tau1_m1, 'tau2_m1:', tau2_m1)
            # IL_rms calculation
            I_L_rms_m1p, x_L_m1p, y_L_m1p = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 0, mode, V1, V2, phi_m1, tau1_m1,
                                                            tau2_m1)
            # debug('y_L_m1p', y_L_m1p)
            # ILc1_rms calculation
            I_Lc1_rms_m1p, x_Lc1_m1p, y_Lc1_m1p = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 1, mode, V1, V2, phi_m1, tau1_m1,
                                                                  tau2_m1)
            # debug('y_Lc1_m1p', y_Lc1_m1p)
            # ILc2_rms calculation
            I_Lc2_rms_m1p, x_Lc2_m1p, y_Lc2_m1p = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 2, mode, V1, V2, phi_m1, tau1_m1,
                                                                  tau2_m1)
            # debug('y_Lc2_m1p', y_Lc2_m1p)
            Irms_Calc.Irms_validation_Gecko(V1.item(), V2.item(), n, Ls, Lc1, Lc2, phi_m1.item(), tau1_m1.item(),
                                            tau2_m1.item(), y_L_m1p.flatten()[0])
            Irms_Calc.plot_Irms(x_L_m1p, y_L_m1p, x_Lc1_m1p, y_Lc1_m1p, x_Lc2_m1p, y_Lc2_m1p, V1.item(), V2.item(),
                                P.item(), mode)

        # # mode 2
        if 0 >= phi >= (tau2 - tau1):
            mode = 2
            # debug('phi_m2:', phi_m2, 'tau1_m2:', tau1_m2, 'tau2_m2:', tau2_m2)
            # IL_rms calculation
            I_L_rms_m2, x_L_m2, y_L_m2 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 0, mode, V1, V2, phi_m2, tau1_m2, tau2_m2)
            # debug('y_L_m2', y_L_m2)
            # ILc1_rms calculation
            I_Lc1_rms_m2, x_Lc1_m2, y_Lc1_m2 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 1, mode, V1, V2, phi_m2, tau1_m2,
                                                               tau2_m2)
            # debug('y_Lc1_m2', y_Lc1_m2)
            # ILc2_rms calculation
            I_Lc2_rms_m2, x_Lc2_m2, y_Lc2_m2 = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 2, mode, V1, V2, phi_m2, tau1_m2,
                                                               tau2_m2)
            # debug('y_Lc2_m2', y_Lc2_m2)
            Irms_Calc.Irms_validation_Gecko(V1.item(), V2.item(), n, Ls, Lc1, Lc2, phi_m2.item(), tau1_m2.item(),
                                            tau2_m2.item(), y_L_m2.flatten()[0])
            Irms_Calc.plot_Irms(x_L_m2, y_L_m2, x_Lc1_m2, y_Lc1_m2, x_Lc2_m2, y_Lc2_m2, V1.item(), V2.item(), P.item(),
                                mode)

    # # IL_rms calculation
    # I_L_rms_m1n, x_L_m1n, y_L_m1n = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 0, -1, V1, V2,
    #                                                 phi_m1_n, tau1_m1_n, tau2_m1_n)
    #
    # # ILc1_rms calculation
    # I_Lc1_rms_m1n, x_Lc1_m1n, y_Lc1_m1n = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 1, -1, V1, V2,
    #                                                       phi_m1_n, tau1_m1_n, tau2_m1_n)
    #
    # # ILc2_rms calculation
    # I_Lc2_rms_m1n, x_Lc2_m1n, y_Lc2_m1n = Irms_Calc.IrmsU(n, Ls, Lc1, Lc2, fs, 2, -1, V1, V2,
    #                                                       phi_m1_n, tau1_m1_n, tau2_m1_n)

    # Ilstart = np.nan_to_num(y_L_m1p, 0) + np.nan_to_num(y_L_m2, 0) #+ np.nan_to_num(y_L_m1n, 0)
    # Irms_Calc.Irms_validation_Gecko(V1[0][0][0], V2[0][0][0], n, Ls, Lc1, Lc2, phi[0][0][0], tau1[0][0][0], tau2[0][0][0], Ilstart[0][0][0][0])

    # if mode == -1:
    #     print('mode:', mode)
    #     Irms_Calc.plot_Irms(x_L_m1n, y_L_m1n, x_Lc1_m1n, y_Lc1_m1n, x_Lc2_m1n, y_Lc2_m1n, V1[0][0][0], V2[0][0][0],
    #                         P[0][0][0], mode)
    # if mode == 1:
    #     print('mode:', mode)
    #     Irms_Calc.plot_Irms(x_L_m1p, y_L_m1p, x_Lc1_m1p, y_Lc1_m1p, x_Lc2_m1p, y_Lc2_m1p, V1[0][0][0], V2[0][0][0],
    #                         P[0][0][0], mode)
    # if mode == 2:
    #     print('mode:', mode)
    #     Irms_Calc.plot_Irms(x_L_m2, y_L_m2, x_Lc1_m2, y_Lc1_m2, x_Lc2_m2, y_Lc2_m2, V1[0][0][0], V2[0][0][0],
    #                         P[0][0][0], mode)
    #===================================================

    # cost function I_HF1^2 + I_HF2^2

    # I_cost = Irms_Calc.I_cost(n, Ls, Lc1, Lc2, fs, V1, V2, phi_m1, tau1_m1, tau2_m1, phi_m2, tau1_m2, tau2_m2)
    # # print(f'{I_cost=}')

    # print(f'{phi*57.29}{tau1*57.29}{tau2*57.29}')

    if tau2 >= phi >= (np.pi - tau1):
        mode = 1
    if 0 >= phi >= (tau2 - tau1):
        mode = 2

    if not (np.isnan(phi.flatten()[0]) or np.isnan(tau1.flatten()[0]) or np.isnan(tau2.flatten()[0])):
        print(f'------------\nV1={V1.flatten()[0]} V, V2={V2.flatten()[0]} V, P={P.flatten()[0]} W, {mode=}')
        print(f'{(phi.flatten()[0] * 57.29):.5f}    {(tau1.flatten()[0] * 57.29):.5f}    {(tau2.flatten()[0] * 57.29):.5f}')

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
    # ## ZVS coverage based on calculation
    # da_mod_results[MOD_KEYS[7]] = np.count_nonzero(zvs) / np.size(zvs)
    # ## ZVS coverage based on calculation
    # da_mod_results[MOD_KEYS[8]] = np.count_nonzero(zvs[~np.isnan(tau1)]) / np.size(zvs[~np.isnan(tau1)])
    # ## Irms
    # da_mod_results[MOD_KEYS[9]] = I_cost
    # # da_mod_results[MOD_KEYS[10]]= error
    return da_mod_results


def objective(trial):
    # Set the basic DAB Specification
    dab = ds.DAB_Data()

    dab.V1_min = 70
    dab.V1_nom = 70
    dab.V1_max = 70
    dab.V1_step = 1

    dab.V2_min = 15
    dab.V2_nom = 20
    dab.V2_max = 30
    dab.V2_step = 3

    dab.P_min = 0
    dab.P_nom = 15
    dab.P_max = 30
    dab.P_step = 3

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

    # dab.Ls = trial.suggest_float("dab.Ls", 120e-6, 125e-6)
    # dab.n = trial.suggest_float("dab.n", 4.2, 4.2)
    # dab.Lc1 = trial.suggest_float("dab.Lc1", 120e-7, 120e-4)
    # dab.Lc2 = trial.suggest_float("dab.Lc2", 120e-7, 120e-4)

    # dab.n = 4.178
    # dab.Ls = 115.6e-6
    # dab.Lc1 = 619e-6
    # dab.Lc2 = 639.4e-6 / (dab.n ** 2)

    dab.n = 4.2
    dab.Ls = 120e-6
    dab.Lc1 = 593e-6
    dab.Lc2 = 680e-6 / (dab.n ** 2)

    da_mod = calc_modulation(dab.n,
                             dab.Ls,
                             dab.Lc1,
                             dab.Lc2,
                             dab.fs,
                             dab['coss_' + mosfet1],
                             dab['coss_' + mosfet2],
                             dab.mesh_V1,
                             dab.mesh_V2,
                             dab.mesh_P, False)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    zvs_coverage = np.count_nonzero(dab.mod_zvs_mask_zvs) / np.size(dab.mod_zvs_mask_zvs)
    I_cost = np.mean(dab.mod_zvs_I_rms_cost[~np.isnan(dab.mod_zvs_I_rms_cost)])
    # debug('Mean', Mean)
    # return zvs_coverage, Mean

    # Introduce the factor
    factored_zvs_coverage = 100 * zvs_coverage
    factored_I_cost = 1 * I_cost

    print(f'{factored_I_cost=}')

    # # Store the original zvs_coverage as a user attribute
    # trial.set_user_attr("original_zvs_coverage", zvs_coverage)
    # # Convert I_Mean to list before setting it as a user attribute
    # trial.set_user_attr("original_Mean", I_Mean.tolist())

    return factored_zvs_coverage, factored_I_cost


def Single_point_validation(x, y, z, gecko: bool):
    # Set the basic DAB Specification
    dab = ds.DAB_Data()
    dab.V1_nom = x
    dab.V1_min = x
    dab.V1_max = x
    dab.V1_step = 1
    dab.V2_nom = y
    dab.V2_min = y
    dab.V2_max = y
    dab.V2_step = 1
    dab.P_min = z
    dab.P_max = z
    dab.P_nom = z
    dab.P_step = 1
    dab.fs = 200000
    dab.Lm = 595e-6

    dab.n = 4.238
    dab.Ls = 132.8e-6
    dab.Lc1 = 619e-6
    dab.Lc2 = 660.1e-6 / (dab.n ** 2)

    # dab.Ls = 125.40e-6
    # dab.n = 4.2
    # dab.Lc1 = 657.1e-6
    # dab.Lc2 = 648.57e-6 / (4.2 ** 2)

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
                             gecko)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    # debug(dab.mod_zvs_phi)
    zvs_coverage = np.count_nonzero(dab.mod_zvs_mask_zvs) / np.size(dab.mod_zvs_mask_zvs)
    # print(f'zvs_coverage: {zvs_coverage}')
    # Mean = dab.mod_zvs_I_rms_cost
    # print(f'I_cost: {Mean}')
    return zvs_coverage


def optimal_zvs_coverage():
    # Set the basic DAB Specification
    dab = ds.DAB_Data()
    dab.V1_nom = 700
    dab.V1_min = 690
    dab.V1_max = 710
    dab.V1_step = 3
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = 3
    # dab.V2_step = 4
    dab.P_min = 0
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 3
    dab.fs = 200000
    dab.Lm = 595e-6

    dab.Ls = 122e-6
    dab.n = 4.2
    dab.Lc1 = 122e-5
    dab.Lc2 = 122e-5

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
                             dab.mesh_P)
    # Unpack the results
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    # debug(dab.mod_zvs_phi)
    zvs_coverage = np.count_nonzero(dab.mod_zvs_mask_zvs) / np.size(dab.mod_zvs_mask_zvs)
    print(f'n: {dab.n}, Lc1: {dab.Lc1}, Lc2: {dab.Lc2}, zvs_coverage: {zvs_coverage}')
    # Mean = dab.mod_zvs_I_rms_Mean
    I_cost = np.mean(dab.mod_zvs_I_rms_cost[~np.isnan(dab.mod_zvs_I_rms_cost)])
    # Mean = np.nanmean(dab.mod_zvs_I_rms_cost)
    print(f'I_cost: {I_cost}')


def proceed_study(study_name: str, number_trials: int) -> None:
    """Proceed with an Optuna study stored in the specified database format.

    :param study_name: Name of the study
    :type study_name: str
    :param number_trials: Number of trials to run
    :type number_trials: int
    """
    # Define the storage path based on the storage type
    storage_path = f"sqlite:///../results/study_{study_name}.sqlite3"

    # Set logging verbosity to show only errors
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Define optimization directions
    directions = ["maximize", "minimize"]

    sampler = optuna.samplers.NSGAIIISampler()

    # Load or create the study in storage
    study_in_storage = optuna.create_study(study_name=study_name,
                                           storage=storage_path,
                                           directions=directions,
                                           load_if_exists=True,
                                           sampler=sampler)

    # Create an in-memory study
    study_in_memory = optuna.create_study(directions=directions, sampler=sampler)

    # Add trials from the storage study to the in-memory study
    study_in_memory.add_trials(study_in_storage.trials)

    # Optimize the in-memory study
    study_in_memory.optimize(objective, n_trials=number_trials, show_progress_bar=True)

    # Add new trials to the storage study
    study_in_storage.add_trials(study_in_memory.trials[-number_trials:])

    # Current timestamp
    timestamp = datetime.now().strftime("%m-%d__%H-%M")

    # Create a unique filename for the Pareto front plot
    filename = f"Pareto_Front__Trials-{len(study_in_storage.trials)}__{timestamp}.html"

    # Specify the directory to save the file
    save_dir = '../results/optuna'
    os.makedirs(save_dir, exist_ok=True)

    # Combine directory and filename
    file_path = os.path.join(save_dir, filename)

    # Plot the Pareto front using the original values and save it to an HTML file
    fig = optuna.visualization.plot_pareto_front(
        study_in_storage,
        target_names=["zvs_coverage / %", "I_cost"])

    # Show and save the plot
    fig.show()
    fig.write_html(file_path)

    # Print the file path for reference
    print('file_path:', file_path)


#===================================
# result of 200k trails:
# Ls = 124.7e-6 / Lc1 = 674.8e-6 / Lc2 = 37.9e-6 -> Lc2_ = 668.5e-6
# at trail number: 93393 / ZVS coverage = 100 / I_cost = 79.6

# final FEM results
# Lc1 = 657.1e-6
# Ls = 125.40e-6
# Lc2_ = 648.57e-6

# old transformer -- measurement results
# n = 4.178
# Ls = 115.6e-6
# Lc1 = 619e-6
# Lc2 = 639.4e-6

# new transformer -- measurement results
# n = 4.238
# Ls = 132.8e-6
# Lc1 = 619e-6
# Lc2 = 660.1e-6

#===================================
# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start.........")

    # # for v2 in range(20, 120, 10):
    for p in range(900, 2000, 200):
        Single_point_validation(700, 200, p, False)

    # Single_point_validation(700, 200, 900, True)
    # objective(1)

    # studyname = datetime.now().strftime("%d%m")
    # proceed_study(study_name=studyname, number_trials=500)

    # optimal_zvs_coverage()

    # Fixed value for v1
    #Irms_Calc.Irms_validation_Gecko(710, 250, 4, 120e-6, 670e-6, 40e-6, 1.6, 0.7, 0.5, 1.5)
    # Single_point_validation(700, 235, 2200)
    # v1 = 700  # [690, 700, 710]
    # v2_values = [175, 235, 295]
    # for v2 in v2_values:
    #     for p in range(200, 2201, 500):
    #         Single_point_validation(v1, v2, p)
    # database_url = 'file:///C:/Users/vijay/Desktop/UPB/Thesis/dab_optimizer/results/optuna/Pareto_Front__Trials-5000__07-18__07-13.html'
    # loaded_study = optuna.create_study(study_name=studyname, storage=database_url, load_if_exists=True)
    # df = loaded_study.trials_dataframe()
    # df.to_csv(f'{studyname}.csv')

    #=========================================================================
    # # Number of trials to run
    # number_of_trials = 3000
    # sampler = optuna.samplers.NSGAIIISampler()
    # # Define storage for Optuna
    # study_name = "example_study"
    # storage_path = f"sqlite:///study_{study_name}.sqlite3"
    # # Create a study with the specified storage
    # study = optuna.create_study(study_name=study_name,
    #                             directions=["maximize", "minimize"],
    #                             storage=storage_path,
    #                             load_if_exists=True,
    #                             sampler=sampler)
    # # Optimize the study
    # study.optimize(objective, n_trials=number_of_trials)
    # # Current timestamp
    # timestamp = datetime.now().strftime("%m-%d__%H-%M")
    # # Create a unique filename based on filter values and number of trials
    # filename = f"Pareto_Front__Trials-{len(study.trials)}__{timestamp}.html"
    # # Specify the directory to save the file
    # save_dir = '../results/optuna'
    # os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    # # Combine directory and filename
    # file_path = os.path.join(save_dir, filename)
    # # Plot the Pareto front using the original values and save it to an HTML file
    # fig = optuna.visualization.plot_pareto_front(
    #     study,
    #     target_names=["zvs_coverage", "I_rms_Mean"],
    #     targets=lambda t: (t.user_attrs["original_zvs_coverage"], t.user_attrs["original_Mean"])
    # )
    # # Add annotations for zvs_threshold, I_rms_threshold, and number of trials
    # fig.add_annotation(
    #     text=f"zvs_factor: 1 - Max <br> current factor: 1 - Min",
    #     xref="paper", yref="paper",
    #     x=0.5, y=1, showarrow=False,
    #     xanchor='center', yanchor='top',
    #     font=dict(size=12)
    # )
    # fig.show()
    # # Save the original Pareto front plot as an HTML file
    # fig.write_html(file_path)
    # print('file_path:', file_path)
    #=========================================================================
    # # Define thresholds for filtering trials
    # zvs_threshold = 0.98
    # I_rms_threshold = 100
    #
    # # Filter the trials based on the given criteria
    # filtered_trials = [
    #     trial for trial in study.trials
    #     if trial.values and trial.values[0] > zvs_threshold and trial.values[1] < I_rms_threshold
    # ]
    #
    # # Check if there are any filtered trials
    # if not filtered_trials:
    #     print("------------- No valid trials found. Please change the trial numbers or filter parameters -------------")
    # else:
    #     # Extract data from the filtered trials
    #     filtered_data = [
    #         {
    #             'trial_number': trial.number,
    #             'zvs_coverage': trial.user_attrs["original_zvs_coverage"],
    #             'I_rms_Mean': trial.user_attrs["original_Mean"],
    #             'params': trial.params
    #         }
    #         for trial in filtered_trials
    #     ]
    #
    # Create a unique filename based on filter values, number of trials, and timestamp
    # timestamp = datetime.now().strftime("%m-%d__%H-%M")
    # filename = f"ParetoFront_ZVSFactor_1_IFacotr_1_{timestamp}.html"

    #
    #     # Output the filename for reference
    #     print(f"Pareto front saved to {filename}")
    #
    # print("Study data stored in SQLite database.")
    #===================

    # # =====================================
    # Trails = 50
    #
    # # Define storage for Optuna
    # study_name = "example_study"
    # storage_path = f"sqlite:///study_{study_name}.sqlite3"
    # study = optuna.create_study(directions=["maximize", "minimize"], storage=storage_path)
    #
    # # Optimize the study
    # study.optimize(objective, n_trials=Trails)
    #
    # # Define thresholds for filtering trials
    # zvs_threshold = 0.98
    # I_rms_threshold = 100
    #
    # # Filter the trials based on the given criteria
    # filtered_trials = [
    #     trial for trial in study.trials
    #     if trial.values and trial.values[0] > zvs_threshold and trial.values[1] < I_rms_threshold
    # ]
    #
    # # Check if there are any filtered trials
    # if not filtered_trials:
    #     print("------------- No valid trials found. Please change the trial numbers or filter parameters -------------")
    # else:
    #     # Extract data from the filtered trials
    #     filtered_data = [
    #         {
    #             'trial_number': trial.number,
    #             'zvs_coverage': trial.user_attrs["original_zvs_coverage"],
    #             'I_rms_Mean': trial.user_attrs["original_Mean"],
    #             'params': trial.params
    #         }
    #         for trial in filtered_trials
    #     ]
    #
    #     # Create a unique filename based on filter values, number of trials, and timestamp
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    #     filename = f"pareto_front_zvs_{zvs_threshold}_I_rms_{I_rms_threshold}A_trials_{len(study.trials)}_{timestamp}.html"
    #
    #     # Plot the Pareto front using the original values and save it to an HTML file
    #     fig = optuna.visualization.plot_pareto_front(
    #         study,
    #         target_names=["zvs_coverage", "I_rms_Mean"],
    #         targets=lambda t: (t.user_attrs["original_zvs_coverage"], t.user_attrs["original_Mean"])
    #     )
    #     fig.write_html(filename)
    #
    #     # Output the filename for reference
    #     print(f"Pareto front saved to {filename}")
    #
    # print("Study data stored in SQLite database.")
    # #=====================================

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
    #         error = Single_point_validation(v1, v2, p)
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

    # Trails = 5000
    # # "maximize", "minimize"
    # study = optuna.create_study(directions=["maximize", "minimize"])
    # study.optimize(objective, n_trials=Trails)
    #
    # # Filter the trials based on the given criteria
    # zvs_threshold = 0.99
    # I_rms_threshold = 90
    # filtered_trials = [
    #     trial for trial in study.trials
    #     if trial.values and trial.values[0] > zvs_threshold and trial.values[1] < I_rms_threshold
    # ]
    #
    # # Check if there are any filtered trials
    # if not filtered_trials:
    #     print("------------- No valid trials found. Please change the trial numbers or filter parameters -------------")
    # # Extract data from the filtered trials
    # filtered_data = [
    #     {
    #         'trial_number': trial.number,
    #         'zvs_coverage': trial.user_attrs["original_zvs_coverage"],
    #         'I_rms_Mean': trial.user_attrs["original_Mean"],
    #         'params': trial.params
    #     }
    #     for trial in filtered_trials
    # ]
    #
    # # To display the original Pareto front
    # # fig = optuna.visualization.plot_pareto_front(study, target_names=["zvs_coverage", "I_rms_Mean"])
    # fig = optuna.visualization.plot_pareto_front(
    #     study,
    #     target_names=["zvs_coverage", "I_cost"],
    #     targets=lambda t: (t.user_attrs["original_zvs_coverage"], t.user_attrs["original_Mean"])
    # )
    # # # Add annotations for zvs_threshold, I_rms_threshold, and number of trials
    # # fig.add_annotation(
    # #     text=f"zvs_threshold: {zvs_threshold}<br>I_rms_threshold: {I_rms_threshold}<br>n_trials: {len(study.trials)}",
    # #     xref="paper", yref="paper",
    # #     x=0.5, y=1, showarrow=False,
    # #     xanchor='center', yanchor='top',
    # #     font=dict(size=12)
    # # )
    # fig.show()
    #
    # # Displaying the filtered trial data
    # for data in filtered_data:
    #     print(
    #         f"Trial Number: {data['trial_number']}, zvs_coverage: {data['zvs_coverage']}, I_rms_Mean: {data['I_rms_Mean']}, Params: {data['params']}")
    #
    # # Current timestamp
    # timestamp = datetime.now().strftime("%m-%d__%H-%M")
    # # Create a unique filename based on filter values and number of trials
    # filename = f"Pareto_Front__Trials-{len(study.trials)}__{timestamp}.html"
    # # Specify the directory to save the file
    # save_dir = r"C:\Users\vijay\Desktop\UPB\Thesis\dab_optimizer\results\optuna"
    # os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    # # Combine directory and filename
    # file_path = os.path.join(save_dir, filename)
    # # Save the original Pareto front plot as an HTML file
    # fig.write_html(file_path)
    # print('file_path:', file_path)
    # # ================================f
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
