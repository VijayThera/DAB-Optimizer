import numpy as np
import pygeckocircuits2 as lpt
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


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

    if mode == -1:
        t0 = np.full_like(phi, 0)
        t1 = (np.pi - tau1)
        t2 = (phi + np.pi)
        t3 = (2 * np.pi + phi - tau2)
        t4 = np.full_like(phi, np.pi)
        t5 = t1 + t4
        t6 = t2 + t4
        t7 = t3 + t4
        t8 = np.full_like(phi, 2*np.pi)

        if IL == 0:
            i_t0 = iL_factor * (d * (np.pi + phi - tau2 * 0.5) - tau1 * 0.5)
            i_t1 = iL_factor * (d * (np.pi + phi - tau2 * 0.5) - tau1 * 0.5)
            i_t2 = iL_factor * (phi + tau1 * 0.5 - d * tau2 * 0.5)
            i_t3 = iL_factor * (np.pi - tau2 + phi + 0.5 * tau1 - d * tau2 * 0.5)

        if IL == 1:
            i_t0 = - iLc1_factor * tau1 * 0.5
            i_t1 = - iLc1_factor * tau1 * 0.5
            i_t2 = iLc1_factor * (tau1 * 0.5 + phi)
            i_t3 = iLc1_factor * (np.pi + phi - tau2 + tau1 * 0.5)

        if IL == 2:
            i_t0 = - iLc2_factor * (np.pi + phi - 0.5 * tau2)
            i_t1 = - iLc2_factor * (tau1 + phi - 0.5 * tau2)
            i_t2 = iLc2_factor * (tau2 * 0.5)
            i_t3 = iLc2_factor * (tau2 * 0.5)

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


def Irms_validation_Gecko(v1, v2, n, Ls, Lc1, Lc2, phi, tau1, tau2, i_Ls_start) : # -> float
    simfilepath = '../circuits/DAB_MOSFET_Modulation_v3 - Copy.ipes'
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
    # print("dab_converter.run_simulation"
    dab_converter.run_simulation(save_file=True)
    dab_converter.get_scope_data(node_names=['i_HF1', 'i_HF2', 'i_Ls', 'i_Lc1', 'i_Lc2_'], file_name='scope_data')
    values = dab_converter.get_values(nodes=['i_HF1', 'i_HF2', 'i_Ls', 'i_Lc1', 'i_Lc2_'],
                                      operations=['rms'],
                                      range_start_stop=[simtime_pre + 2e-8, simtime_pre + simtime + 2e-8])

    # i_rms = values['rms']['i_Ls'] + values['rms']['i_Lc1'] + values['rms']['i_Lc2_']
    # print(i_rms)
    # return i_rms


def plot_Irms(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, V1, V2,
              P, mode):
    # Load the CSV files
    file1 = "../circuits/results/v3.csv"
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

    # x0_ = x0 + np.full_like(x0, 5e-6)
    # x0 = np.append(x0, x0_)
    # y0 = np.append(y0, y0)
    #
    # x1_ = x1 + np.full_like(x1, 5e-6)
    # x1 = np.append(x1, x1_)
    # y1 = np.append(y1, y1)
    #
    # x2_ = x2 + np.full_like(x2, 5e-6)
    # x2 = np.append(x2, x2_)
    # y2 = np.append(y2, y2)

    # # Create the plots
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axs[0].plot(x0+x1, y0, 'o-r', ms=3, label='Calc.')
    # axs[0].plot(df['t_shifted'], df['i_HF1'], color='b', label='Gecko')
    # axs[0].set_title('Plot of i_L')
    # axs[0].grid(color='gray', linestyle='--', linewidth=0.5)
    #
    # axs[1].plot(x0-x2, y0, 'o-r', ms=3, label='Calc.')
    # axs[1].plot(df['t_shifted'], df['i_HF2'], color='b', label='Gecko')
    # axs[1].set_title('Plot of i_Lc1')
    # axs[1].grid(color='gray', linestyle='--', linewidth=0.5)
    #
    # # Add a common xlabel and ylabel
    # fig.text(0.5, 0.04, 'Time (Sec)', ha='center', va='center')
    # fig.text(0.04, 0.5, 'Current (A)', ha='center', va='center', rotation='vertical')
    #
    # # Add a common legend
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    # plt.grid(True)
    # ## Maximize the window to fullscreen
    # # plt.get_current_fig_manager().window.showMaximized()
    # # plt.show()

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
    # Plotting the data
    # Create a new folder to save the image
    # Current timestamp
    timestamp = datetime.now().strftime("%m-%d__%H-%M")
    output_folder_ = '../results/currents'
    os.makedirs(output_folder_, exist_ok=True)

    # Define the file name for the image
    image_name_ = f'Irms_Calc_vs_Gecko_{V1:.0f}_{V2:.0f}_{P:.0f}_plot__{timestamp}.png'
    image_path_ = os.path.join(output_folder_, image_name_)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(x0, y0, 'o-r', ms=3, label='Calc.')
    # plt.plot(x_dense_filtered_0, y_dense_filtered_0, 'o-b', ms=3, label='Gecko')
    plt.plot(df['t_shifted'], df['i_Ls'], '-g', label='CSV')
    plt.title('Plot of i_L')
    plt.xlabel('Time (Sec)')
    plt.ylabel('iL (A)')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x1, y1, 'o-r', ms=3, label='Calc.')
    # plt.plot(x_dense_filtered_1, y_dense_filtered_1, 'o-b', ms=3, label='Gecko')
    plt.plot(df['t_shifted'], df['i_Lc1'], '-g', label='CSV')
    plt.title('Plot of i_Lc1')
    plt.xlabel('Time (Sec)')
    plt.ylabel('iLc1 (A)')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(x2, y2, 'o-r', ms=3, label='Calc.')
    # plt.plot(x_dense_filtered_2, y_dense_filtered_2, 'o-b', ms=3, label='Gecko')
    plt.plot(df['t_shifted'], df['i_Lc2_'], '-g', label='CSV')
    plt.title('Plot of i_Lc2')
    plt.xlabel('Time (Sec)')
    plt.ylabel('iLc2 (A)')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

    # Maximize the window to fullscreen
    textstr = f'V1: {V1}V, V2: {V2}V, P: {P}W, Mode: {mode}'
    plt.gcf().text(0.5, 0.95, textstr, fontsize=12, horizontalalignment='center', verticalalignment='top')
    plt.get_current_fig_manager().window.showMaximized()
    # plt.show()
    plt.savefig(image_path_, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {image_path_}")


def I_cost(n, Ls, Lc1, Lc2, fs, V1, V2, phi_m1, tau1_m1, tau2_m1, phi_m2, tau1_m2, tau2_m2) -> [np.ndarray]:

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

    t0_m1 = np.full_like(V1, 0)
    t1_m1 = (np.pi - tau1_m1)
    t2_m1 = phi_m1
    t3_m1 = (np.pi + phi_m1 - tau2_m1)
    t4_m1 = np.full_like(V1, np.pi)
    t5_m1 = t1_m1 + t4_m1
    t6_m1 = t2_m1 + t4_m1
    t7_m1 = t3_m1 + t4_m1
    t8_m1 = np.full_like(V1, 2*np.pi)
    times_m1 = np.array([t0_m1, t1_m1, t2_m1, t3_m1, t4_m1, t5_m1, t6_m1, t7_m1, t8_m1])

    t0_m2 = np.full_like(V1, 0)
    t1_m2 = (np.pi - tau1_m2)
    t2_m2 = (np.pi + phi_m2 - tau2_m2)
    t3_m2 = (np.pi + phi_m2)
    t4_m2 = np.full_like(V1, np.pi)
    t5_m2 = t1_m2 + t4_m2
    t6_m2 = t2_m2 + t4_m2
    t7_m2 = t3_m2 + t4_m2
    t8_m2 = np.full_like(V1, 2 * np.pi)
    times_m2 = np.array([t0_m2, t1_m2, t2_m2, t3_m2, t4_m2, t5_m2, t6_m2, t7_m2, t8_m2])

    i_HF1_m1_t0 = -iL_factor * (d * (phi_m1 - tau2_m1 * 0.5) + tau1_m1 * 0.5) - iLc1_factor * tau1_m1 * 0.5
    i_HF1_m1_t1 = iL_factor * (d * (-tau1_m1 + tau2_m1 * 0.5 - phi_m1 + np.pi) - tau1_m1 * 0.5) - iLc1_factor * tau1_m1 * 0.5
    i_HF1_m1_t2 = -iL_factor * (-d * tau2_m1 * 0.5 - tau1_m1*0.5 - phi_m1 + np.pi) + iLc1_factor * (tau1_m1 * 0.5 + phi_m1 - np.pi)
    i_HF1_m1_t3 = iL_factor * (d * tau2_m1 * 0.5 + tau1_m1 * 0.5 - tau2_m1 + phi_m1) + iLc1_factor * (tau1_m1 * 0.5 - tau2_m1 + phi_m1)
    i_HF1_m1_t4 = -i_HF1_m1_t0
    i_HF1_m1_t5 = -i_HF1_m1_t1
    i_HF1_m1_t6 = -i_HF1_m1_t2
    i_HF1_m1_t7 = -i_HF1_m1_t3
    i_HF1_m1_t8 = i_HF1_m1_t0
    currents_HF1_m1 = np.array([i_HF1_m1_t0, i_HF1_m1_t1, i_HF1_m1_t2, i_HF1_m1_t3, i_HF1_m1_t4, i_HF1_m1_t5, i_HF1_m1_t6, i_HF1_m1_t7, i_HF1_m1_t8])
    i_HF1_m1 = integral(times_m1, currents_HF1_m1)
    # print(f'i_HF1_m1:{i_HF1_m1}')

    i_HF1_m2_t0 = iL_factor * (d * tau2_m2 * 0.5 - tau1_m2 * 0.5) - iLc1_factor * tau1_m2 * 0.5
    i_HF1_m2_t1 = iL_factor * (d * tau2_m2 * 0.5 - tau1_m2 * 0.5) - iLc1_factor * tau1_m2 * 0.5
    i_HF1_m2_t2 = iL_factor * (d * tau2_m2 * 0.5 + tau1_m2 * 0.5 + phi_m2 - tau2_m2) + iLc1_factor * (tau1_m2 * 0.5 - tau2_m2 + phi_m2)
    i_HF1_m2_t3 = iL_factor * (phi_m2 - d * tau2_m2 * 0.5 + tau1_m2 * 0.5) + iLc1_factor * (tau1_m2 * 0.5 + phi_m2)
    i_HF1_m2_t4 = -i_HF1_m2_t0
    i_HF1_m2_t5 = -i_HF1_m2_t1
    i_HF1_m2_t6 = -i_HF1_m2_t2
    i_HF1_m2_t7 = -i_HF1_m2_t3
    i_HF1_m2_t8 = i_HF1_m2_t0
    currents_HF1_m2 = np.array([i_HF1_m2_t0, i_HF1_m2_t1, i_HF1_m2_t2, i_HF1_m2_t3, i_HF1_m2_t4, i_HF1_m2_t5, i_HF1_m2_t6, i_HF1_m2_t7, i_HF1_m2_t8])
    i_HF1_m2 = integral(times_m2, currents_HF1_m2)
    # print(f'i_HF1_m2:{i_HF1_m2}')

    i_HF2_m1_t0 = - iL_factor * (d * (phi_m1 - tau2_m1 * 0.5) + tau1_m1 * 0.5) + iLc2_factor * (tau2_m1 * 0.5 - phi_m1)
    i_HF2_m1_t1 = iL_factor * (d * (-tau1_m1 + tau2_m1 * 0.5 - phi_m1 + np.pi) - tau1_m1 * 0.5) - iLc2_factor * (tau1_m1 - tau2_m1 * 0.5 + phi_m1 - np.pi)
    i_HF2_m1_t2 = -iL_factor * (-d * tau2_m1 * 0.5 - tau1_m1*0.5 - phi_m1 + np.pi) - iLc2_factor * (tau2_m1 * 0.5)
    i_HF2_m1_t3 = iL_factor * (d * tau2_m1 * 0.5 + tau1_m1 * 0.5 - tau2_m1 + phi_m1) - iLc2_factor * (tau2_m1 * 0.5)
    i_HF2_m1_t4 = -i_HF2_m1_t0
    i_HF2_m1_t5 = -i_HF2_m1_t1
    i_HF2_m1_t6 = -i_HF2_m1_t2
    i_HF2_m1_t7 = -i_HF2_m1_t3
    i_HF2_m1_t8 = i_HF2_m1_t0
    currents_HF2_m1 = np.array([i_HF2_m1_t0, i_HF2_m1_t1, i_HF2_m1_t2, i_HF2_m1_t3, i_HF2_m1_t4, i_HF2_m1_t5, i_HF2_m1_t6, i_HF2_m1_t7, i_HF2_m1_t8])
    i_HF2_m1 = integral(times_m1, currents_HF2_m1)
    # print(f'i_HF2_m1:{i_HF2_m1}')

    i_HF2_m2_t0 = iL_factor * (d * tau2_m2 * 0.5 - tau1_m2 * 0.5) + iLc2_factor * tau2_m2 * 0.5
    i_HF2_m2_t1 = iL_factor * (d * tau2_m2 * 0.5 - tau1_m2 * 0.5) + iLc2_factor * tau2_m2 * 0.5
    i_HF2_m2_t2 = iL_factor * (d * tau2_m2 * 0.5 + tau1_m2 * 0.5 + phi_m2 - tau2_m2) + iLc2_factor * tau2_m2 * 0.5
    i_HF2_m2_t3 = iL_factor * (phi_m2 - d * tau2_m2 * 0.5 + tau1_m2 * 0.5) - iLc2_factor * tau2_m2 * 0.5
    i_HF2_m2_t4 = -i_HF2_m2_t0
    i_HF2_m2_t5 = -i_HF2_m2_t1
    i_HF2_m2_t6 = -i_HF2_m2_t2
    i_HF2_m2_t7 = -i_HF2_m2_t3
    i_HF2_m2_t8 = i_HF2_m2_t0
    currents_HF2_m2 = np.array([i_HF2_m2_t0, i_HF2_m2_t1, i_HF2_m2_t2, i_HF2_m2_t3, i_HF2_m2_t4, i_HF2_m2_t5, i_HF2_m2_t6, i_HF2_m2_t7, i_HF2_m2_t8])
    # print(f'currents_HF2_m2:{currents_HF2_m2}')
    i_HF2_m2 = integral(times_m2, currents_HF2_m2)
    # print(f'i_HF2_m2:{i_HF2_m2}')

    i_HF1_rms_mean = np.nan_to_num(i_HF1_m1, 0)+ np.nan_to_num(i_HF1_m2, 0)
    i_HF2_rms_mean = n * (np.nan_to_num(i_HF2_m1, 0)+ np.nan_to_num(i_HF2_m2, 0))

    # i_HF1_rms_mean = np.nanmean(np.nan_to_num(i_HF1_m1, 0)+ np.nan_to_num(i_HF1_m2, 0))
    # i_HF2_rms_mean = n * np.nanmean(np.nan_to_num(i_HF2_m1, 0)+ np.nan_to_num(i_HF2_m2, 0))

    # print(f'i_HF1_rms_mean: {i_HF1_rms_mean}')
    # print(f'i_HF2_rms_mean: {i_HF2_rms_mean}')

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axs[0].plot(times_m1, currents_HF1_m1, 'o-r', ms=3, label='Calc.')
    # # axs[0].plot(df['t_shifted'], df['i_Ls'], color='b', label='Gecko')
    # axs[0].set_title('Plot of i_HF1')
    # axs[0].grid(color='gray', linestyle='--', linewidth=0.5)
    #
    # axs[1].plot(times_m1, currents_HF2_m1, 'o-r', ms=3, label='Calc.')
    # # axs[1].plot(df['t_shifted'], df['i_Lc1'], color='b', label='Gecko')
    # axs[1].set_title('Plot of i_HF2')
    # axs[1].grid(color='gray', linestyle='--', linewidth=0.5)
    #
    # # Add a common xlabel and ylabel
    # fig.text(0.5, 0.04, 'Time (Sec)', ha='center', va='center')
    # fig.text(0.04, 0.5, 'Current (A)', ha='center', va='center', rotation='vertical')
    #
    # # Add a common legend
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    # plt.grid(True)

    # return np.nansum(i_HF1_m1, i_HF1_m2), np.nansum(i_HF2_m1, i_HF2_m2)
    I_cost_mat = (i_HF1_rms_mean ** 2 + i_HF2_rms_mean ** 2)
    I_cost_mat = np.where(I_cost_mat == 0, np.nan, I_cost_mat)
    # print(I_cost_mat)
    return I_cost_mat


def integral(times, currents):
    I_integrated = np.full_like(times[0], 0)

    for i in range(8):
        I_integrated += (times[i + 1] - times[i]) * 0.33334 * (currents[i] ** 2 + currents[i + 1] ** 2 + currents[i + 1] * currents[i])

    I_rms = np.sqrt(I_integrated / (2 * np.pi))
    # print(I_rms)
    return I_rms
