import numpy as np
import pygeckocircuits2 as lpt


# def Irms(n, Ls, Lc1, Lc2, fs, IL, mode, V1, V2, phi, tau1, tau2) -> [np.ndarray, np.ndarray, np.ndarray]:
#
#     I_integrated = np.zeros_like(phi)
#     i_alpha, i_beta, i_gamma, i_delta, angles, currents, x, y = (np.full_like(phi, np.nan) for _ in range(8))
#     ws = 2 * np.pi * fs
#     Ts = 1/fs
#     iL_factor = V1 / (ws * Ls)
#     iLc1_factor = V1 / (ws * Lc1)
#     d = n * V2 / V1
#     # Transform Lc2 to side 1
#     Lc2_ = Lc2 * n ** 2
#     # Transform V2 to side 1
#     V2_ = V2 * n
#     iLc2_factor = V2_ / (ws * Lc2_)
#
#     if mode == 1:
#         alpha = 3.14159 - tau1
#         beta = 3.14159 + phi - tau2
#         gamma = np.full_like(phi, 3.14159)
#         delta = 3.14159 + phi
#         alpha_ = 2 * 3.14159 - tau1
#         beta_ = 2 * 3.14159 + phi - tau2
#         gamma_ = np.full_like(phi, 0)
#         delta_ = phi
#         gamma__ = np.full_like(phi, 2 * 3.14159)
#         # t0 = np.full_like(phi, 0)
#         # t1 = (np.pi - tau1)/ws
#         # t2 = phi/ws
#         # t3 = (np.pi + phi - tau2)/ws
#         # t4 = np.full_like(phi, Ts*0.5)
#         # t5 = t1 + t4
#         # t6 = t1 + t4
#         # t7 = t1 + t4
#         # t8 = t1 + t4
#
#         if IL == 0:
#             i_alpha = iL_factor * (d * (-tau1 + tau2 * 0.5 - phi + np.pi) - tau1 * 0.5)
#             i_beta = iL_factor * (d * tau2 * 0.5 + tau1 * 0.5 - tau2 + phi)
#             i_gamma = iL_factor * (d * (-tau2 * 0.5 + phi) + tau1 * 0.5)
#             i_delta = iL_factor * (-d * tau2 * 0.5 - tau1 * 0.5 - phi + np.pi)
#             # i_t0 = -iL_factor * (d * (phi - tau2 * 0.5) + tau1 * 0.5)
#             # i_t1 = iL_factor * (d * (-tau1 + tau2 * 0.5 - phi + np.pi) - tau1 * 0.5)
#             # i_t2 = -iL_factor * (-d * tau2 * 0.5 - tau1*0.5 - phi + np.pi)
#             # i_t3 = iL_factor * (d * tau2 * 0.5 + tau1 * 0.5 - tau2 + phi)
#
#         if IL == 1:
#             i_alpha = - iLc1_factor * tau1 * 0.5
#             i_beta = iLc1_factor * (tau1 * 0.5 - tau2 + phi)
#             i_gamma = iLc1_factor * tau1 * 0.5
#             i_delta = iLc1_factor * (-tau1 * 0.5 - phi + np.pi)
#
#         if IL == 2:
#             i_alpha = iLc2_factor * (tau1 - tau2 * 0.5 + phi - np.pi)
#             i_beta = - iLc2_factor * (tau2 * 0.5)
#             i_gamma = iLc2_factor * (tau2 * 0.5 - phi)
#             i_delta = iLc2_factor * tau2 * 0.5
#
#         i_alpha_ = -i_alpha
#         i_beta_ = -i_beta
#         i_gamma_ = -i_gamma
#         i_delta_ = -i_delta
#         # i_t4 = -i_t0
#         # i_t5 = -i_t1
#         # i_t6 = -i_t2
#         # i_t7 = -i_t3
#         # i_t8 = i_t0
#
#         # Create arrays for angles and currents
#         angles = np.array([gamma_, alpha, delta_, beta, gamma, alpha_, delta, beta_, gamma__])
#         currents = np.array([i_gamma_, i_alpha, i_delta_, i_beta, i_gamma, i_alpha_, i_delta, i_beta_, i_gamma_])
#         # times = np.array([t0, t1, t2, t3, t4, t5, t6, t7, t8])
#         # currents = np.array([i_t0, i_t1, i_t2, i_t3, i_t4, i_t5, i_t6, i_t7, i_t8])
#
#         # Sort the angles and currents according to the angle order
#         sorted_indices = np.argsort(angles, axis=0)
#         x = np.take_along_axis(angles, sorted_indices, axis=0)
#         y = np.take_along_axis(currents, sorted_indices, axis=0)
#
#         for i in range(8):
#             I_integrated += (angles[i + 1] - angles[i]) * 0.33334 * (currents[i] ** 2 + currents[i + 1] ** 2 + currents[i + 1] * currents[i])
#
#     if mode == 2:
#         alpha_ = np.full_like(phi, 0)
#         alpha = 3.14159 - tau1
#         beta = 3.14159 + phi - tau2
#         gamma = np.full_like(phi, 3.14159)
#         delta = 3.14159 + phi
#         gamma_ = 2 * 3.14159 - tau1
#         delta_ = 2 * 3.14159 + phi - tau2
#         beta_ = 2 * 3.14159 + phi
#         alpha__ = np.full_like(phi, 2 * 3.14159)
#
#         if IL == 0:
#             # i_alpha = iL_factor * (d * tau2 * 0.5 - tau1 * 0.5) * 0.5
#             # i_beta = iL_factor * (d * tau2 * 0.5 + tau1 * 0.5 - tau2 + phi) * 1
#             # i_gamma = - iL_factor * (-d * tau2 * 0.5 + tau1 * 0.5) * -0.5
#             # # i_delta = - iL_factor * (d * tau2 * 0.5 - tau1 * 0.5)
#             # i_delta = iL_factor * (-d * tau2 * 0.5 + tau1 + phi) * -0.03
#             # i_alpha_ = -i_alpha * -0.5
#             # i_beta_ = -i_beta * -0.1
#             # i_gamma_ = -i_gamma * -0.9
#             # i_delta_ = -i_beta
#
#             i_t0 = iL_factor * (d*tau2*0.5-tau1*0.5)
#             i_t1 = iL_factor * (d*tau2*0.5-tau1*0.5)
#             i_t2 = iL_factor * (d*tau2*0.5+tau1*0.5+phi-tau2)
#             i_t3 = iL_factor * (phi-d*tau2*0.5+tau1*0.5)
#             i_t4 = -i_t0
#             i_t5 = -i_t1
#             i_t6 = -i_t2
#             i_t7 = -i_t3
#             i_t8 = i_t0
#
#             # Create arrays for angles and currents
#             angles = np.array([alpha_, alpha, beta, delta, gamma, gamma_, delta_, beta_, alpha__])
#             currents = np.array([i_t0, i_t1, i_t2, i_t3, i_t4, i_t5, i_t6, i_t7, i_t8])
#
#         if IL == 1:
#             i_alpha = - iLc1_factor * tau1 * 0.5
#             i_beta = iLc1_factor * (tau1 * 0.5 - tau2 + phi)
#             i_gamma = iLc1_factor * tau1 * 0.5 * 0.9
#             i_delta = iLc1_factor * (tau1 * 0.5 + phi)
#             i_alpha_ = i_alpha
#             i_beta_ = i_beta
#             i_gamma_ = i_gamma
#             i_delta_ = i_delta
#
#             # Create arrays for angles and currents
#             angles = np.array([alpha_, alpha, beta, delta, gamma, gamma_, delta_, beta_, alpha__])
#             currents = np.array([i_alpha_, i_alpha, i_beta, i_delta, i_gamma, i_gamma_, i_delta_, i_beta_, i_alpha_])
#             # angles = np.array([alpha_, alpha, beta, delta, gamma, gamma_, delta_, beta_, alpha__])
#             # currents = np.array([i_alpha_, i_alpha, i_beta, i_delta, i_gamma, i_gamma_, i_delta_, i_beta_, i_alpha_])
#
#         if IL == 2:
#             i_alpha = - iLc2_factor * tau2 * 0.5
#             i_beta = - iLc2_factor * tau2 * 0.5
#             i_gamma = iLc2_factor * tau2 * 0.5
#             i_delta = iLc2_factor * tau2 * 0.5
#             i_alpha_ = i_alpha
#             i_beta_ = i_beta
#             i_gamma_ = i_gamma
#             i_delta_ = i_delta
#
#             # Create arrays for angles and currents
#             angles = np.array([alpha_, alpha, beta, delta, gamma, gamma_, delta_, beta_, alpha__])
#             currents = np.array([i_alpha_, i_alpha, i_beta, i_delta, i_gamma, i_gamma_, i_delta_, i_beta_, i_alpha_])
#             # angles = np.array([alpha_, alpha, beta, delta, gamma, gamma_, delta_, beta_, alpha__])
#             # currents = np.array([i_alpha_, i_alpha, i_beta, i_delta, i_gamma, i_gamma_, i_delta_, i_beta_, i_alpha_])
#
#         # # Sort the angles and currents according to the angle order
#         # sorted_indices = np.argsort(angles, axis=0)
#         # x = np.take_along_axis(angles, sorted_indices, axis=0)
#         # y = np.take_along_axis(currents, sorted_indices, axis=0)
#
#         for i in range(8):
#             # I_integrated += (x[i + 1] - x[i])*0.33334*(y[i]**2 + y[i+1]**2 + y[i + 1]*y[i])
#             I_integrated += (angles[i + 1] - angles[i]) * 0.33334 * (currents[i] ** 2 + currents[i + 1] ** 2 + currents[i + 1] * currents[i])
#
#     I_rms = np.sqrt(I_integrated / (2 * np.pi))
#     return I_rms, angles, currents
#     # return I_rms, x, y


def IrmsU(n, Ls, Lc1, Lc2, fs, IL, mode, V1, V2, phi, tau1, tau2) -> [np.ndarray, np.ndarray, np.ndarray]:

    I_integrated = np.zeros_like(phi)
    t0, t1, t2, t3, t4, t5, t6, t7, t8, times = (np.full_like(phi, np.nan) for _ in range(10))
    i_t0, i_t1, i_t2, i_t3, i_t4, i_t5, i_t6, i_t7, i_t8, currents = (np.full_like(phi, np.nan) for _ in range(10))
    ws = 2 * np.pi * fs
    Ts = 1/fs
    iL_factor = V1 / (ws * Ls)
    iLc1_factor = V1 / (ws * Lc1)
    d = n * V2 / V1
    # Transform Lc2 to side 1
    Lc2_ = Lc2 * n ** 2
    # Transform V2 to side 1
    V2_ = V2 * n
    iLc2_factor = V2_ / (ws * Lc2_)

    if mode == 1:
        t0 = np.full_like(phi, 0)
        t1 = (np.pi - tau1)
        t2 = phi
        t3 = (np.pi + phi - tau2)
        t4 = np.full_like(phi, np.pi)
        t5 = t1 + t4
        t6 = t2 + t4
        t7 = t3 + t4
        t8 = np.full_like(phi, 2*np.pi)

        if IL == 0:
            i_t0 = -iL_factor * (d * (phi - tau2 * 0.5) + tau1 * 0.5)
            i_t1 = iL_factor * (d * (-tau1 + tau2 * 0.5 - phi + np.pi) - tau1 * 0.5)
            i_t2 = -iL_factor * (-d * tau2 * 0.5 - tau1*0.5 - phi + np.pi)
            i_t3 = iL_factor * (d * tau2 * 0.5 + tau1 * 0.5 - tau2 + phi)

        if IL == 1:
            i_t0 = - iLc1_factor * tau1 * 0.5
            i_t1 = - iLc1_factor * tau1 * 0.5
            i_t2 = iLc1_factor * (tau1 * 0.5 + phi - np.pi)
            i_t3 = iLc1_factor * (tau1 * 0.5 - tau2 + phi)

        if IL == 2:
            i_t0 = - iLc2_factor * (tau2 * 0.5 - phi)
            i_t1 = iLc2_factor * (tau1 - tau2 * 0.5 + phi - np.pi)
            i_t2 = - iLc2_factor * (tau2 * 0.5)
            i_t3 = - iLc2_factor * (tau2 * 0.5)

    if mode == 2:
        t0 = np.full_like(phi, 0)
        t1 = (np.pi - tau1)
        t2 = (np.pi + phi - tau2)
        t3 = (np.pi + phi)
        t4 = np.full_like(phi, np.pi)
        t5 = t1 + t4
        t6 = t2 + t4
        t7 = t3 + t4
        t8 = np.full_like(phi, 2*np.pi)

        if IL == 0:
            i_t0 = iL_factor * (d*tau2*0.5-tau1*0.5)
            i_t1 = iL_factor * (d*tau2*0.5-tau1*0.5)
            i_t2 = iL_factor * (d*tau2*0.5+tau1*0.5+phi-tau2)
            i_t3 = iL_factor * (phi-d*tau2*0.5+tau1*0.5)

        if IL == 1:
            i_t0 = - iLc1_factor * tau1 * 0.5
            i_t1 = - iLc1_factor * tau1 * 0.5
            i_t2 = iLc1_factor * (tau1 * 0.5 - tau2 + phi)
            i_t3 = iLc1_factor * (tau1 * 0.5 + phi)

        if IL == 2:
            i_t0 = - iLc2_factor * tau2 * 0.5
            i_t1 = - iLc2_factor * tau2 * 0.5
            i_t2 = - iLc2_factor * tau2 * 0.5
            i_t3 = iLc2_factor * tau2 * 0.5

    i_t4 = -i_t0
    i_t5 = -i_t1
    i_t6 = -i_t2
    i_t7 = -i_t3
    i_t8 = i_t0

    times = np.array([t0, t1, t2, t3, t4, t5, t6, t7, t8])
    currents = np.array([i_t0, i_t1, i_t2, i_t3, i_t4, i_t5, i_t6, i_t7, i_t8])
    for i in range(8):
        I_integrated += (times[i + 1] - times[i]) * 0.33334 * (currents[i] ** 2 + currents[i + 1] ** 2 + currents[i + 1] * currents[i])

    I_rms = np.sqrt(I_integrated / (2*np.pi))
    return I_rms, times, currents


def Irms_validation_Gecko(v1, v2, n, Ls, Lc1, Lc2, phi, tau1, tau2, i_Ls_start) -> float:
    simfilepath = '../circuits/DAB_MOSFET_Modulation_v3 - copy.ipes'
    timestep = 1e-9
    simtime = 10e-6
    timestep_pre = 10e-9
    simtime_pre = 10e-3

    # Radians to Degrees
    phi = np.around(phi * 180 / 3.14159, 5)
    tau1 = np.around(tau1 * 180 / 3.14159, 5)
    tau2 = np.around(tau2 * 180 / 3.14159, 5)

    # print("lpt.GeckoSimulation(simfilepath)")
    dab_converter = lpt.GeckoSimulation(simfilepath, simtime=simtime, timestep=timestep, simtime_pre=simtime_pre,
                                        timestep_pre=timestep_pre)
    # dab_converter.get_global_parameters(['phi', 'tau1', 'tau2', 'v_dc1', 'v_dc2', 'f_s', 'Lc1', 'Lc2', 'Ls', 'i_Ls_start'])
    params = {'n': n, 'v_dc1': v1, 'v_dc2': v2, 'f_s': 200000, 't_dead1': 100e-9, 't_dead2': 100e-9,
              'Ls': Ls, 'i_Ls_start': i_Ls_start, 'Lc1': Lc1, 'Lc2': Lc2,
              'phi': phi, 'tau1': tau1, 'tau2': tau2}
    # print(params)
    dab_converter.set_global_parameters(params)
    # print("dab_converter.run_simulation")
    dab_converter.run_simulation(save_file=True)
    dab_converter.get_scope_data(node_names=['i_Ls', 'i_Lc1', 'i_Lc2_'], file_name='scope_data')
    values = dab_converter.get_values(nodes=['i_Ls', 'i_Lc1', 'i_Lc2_'],
                                      operations=['rms'],
                                      range_start_stop=[simtime_pre + 2e-8, simtime_pre + simtime + 2e-8])

    i_rms = values['rms']['i_Ls'] + values['rms']['i_Lc1'] + values['rms']['i_Lc2_']
    # print(i_rms)
    return i_rms
