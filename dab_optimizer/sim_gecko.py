#!/usr/bin/python3
# coding: utf-8
# python >= 3.10

import numpy as np

import leapythontoolbox as lpt
import classes_datasets as ds
import debug_tools as db


@db.timeit
def start_sim(mesh_V1, mesh_V2, mesh_P, mvvp_phi, mvvp_tau1, mvvp_tau2):
    # Gecko Basics
    # TODO make this variable
    sim_filepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC.ipes'
    if not __debug__:
        dab_converter = lpt.GeckoSimulation(sim_filepath)

    # init array to store RMS currents
    mvvp_iLs = np.full_like(mesh_V1, np.nan)
    print(mvvp_iLs.shape)
    mvvp_S11_p_sw = np.full_like(mesh_V1, np.nan)

    for vec_vvp in np.ndindex(mvvp_iLs.shape):
        # print(vec_vvp, mvvp_phi[vec_vvp], mvvp_tau1[vec_vvp], mvvp_tau2[vec_vvp], sep='\n')

        # set simulation parameters and convert tau to inverse-tau for Gecko
        sim_params = {
            # TODO find a way to do this with sparse arrays
            'v_dc1': mesh_V1[vec_vvp].item(),
            'v_dc2': mesh_V2[vec_vvp].item(),
            'phi': mvvp_phi[vec_vvp].item() / np.pi * 180,
            'tau1_inv': (np.pi - mvvp_tau1[vec_vvp].item()) / np.pi * 180,
            'tau2_inv': (np.pi - mvvp_tau2[vec_vvp].item()) / np.pi * 180
        }
        print(sim_params)
        # print("phi: ", type(sim_params['phi']))

        # start simulation for this operation point
        # TODO optimize for multithreading, maybe multiple Gecko instances needed
        if not __debug__:
            dab_converter.set_global_parameters(sim_params)
        # TODO time settings should be variable
        # dab_converter.run_simulation(timestep=100e-12, simtime=15e-6, timestep_pre=50e-9, simtime_pre=10e-3)
        # TODO Bug in LPT with _pre settings
        # does this still run a pre-simulation like in the model?
        if not __debug__:
            dab_converter.run_simulation(timestep=100e-12, simtime=15e-6)
            values_mean = dab_converter.get_values(
                nodes=['p_dc1', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond'],
                operations=['mean']
            )
            values_rms = dab_converter.get_values(
                nodes=['i_Ls'],
                operations=['rms']
            )
        else:
            # generate some fake data for debugging
            values_mean = {'mean': {'p_dc1': np.random.uniform(0.0, 1000),
                                    'S11_p_sw': np.random.uniform(0.0, 10),
                                    'S11_p_cond': np.random.uniform(0.0, 10),
                                    'S12_p_sw': np.random.uniform(0.0, 1000),
                                    'S12_p_cond': np.random.uniform(0.0, 100)}}
            values_rms = {'rms': {'i_Ls': np.random.uniform(0.0, 10)}}

        power_deviation = mesh_P[vec_vvp].item() and values_mean['mean']['p_dc1'] / mesh_P[vec_vvp].item()
        print("power_target", mesh_P[vec_vvp].item())
        print("power_deviation", power_deviation)

        # save simulation results in array
        mvvp_iLs[vec_vvp] = values_rms['rms']['i_Ls']
        mvvp_S11_p_sw[vec_vvp] = values_mean['mean']['S11_p_sw']

    return mvvp_iLs, mvvp_S11_p_sw


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module SIM ...")
