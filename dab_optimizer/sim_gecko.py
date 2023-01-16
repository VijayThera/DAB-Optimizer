#!/usr/bin/python3
# coding: utf-8
# python >= 3.10

import numpy as np

import leapythontoolbox as lpt
import classes_datasets as ds
from debug_tools import *


@timeit
def start_sim(mesh_V1: np.ndarray, mesh_V2: np.ndarray, mesh_P: np.ndarray,
              mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray) -> dict:
    # Gecko Basics
    # TODO make this variable
    sim_filepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC.ipes'
    if not __debug__:
        dab_converter = lpt.GeckoSimulation(sim_filepath)

    # mean values we want to get from the simulation
    l_means = ['p_dc1', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond']
    # init array to store RMS currents
    mvvp_iLs = np.full_like(mesh_V1, np.nan)
    # print(mvvp_iLs.shape)
    # ['p_dc1', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond']
    # mvvp_p_dc1 = np.full_like(mesh_V1, np.nan)
    # mvvp_S11_p_sw = np.full_like(mesh_V1, np.nan)
    # mvvp_S11_p_cond = np.full_like(mesh_V1, np.nan)
    # mvvp_S12_p_sw = np.full_like(mesh_V1, np.nan)
    # mvvp_S12_p_cond = np.full_like(mesh_V1, np.nan)
    d_mvvp_means = dict()
    for k in l_means:
        d_mvvp_means[k] = np.full_like(mesh_V1, np.nan)

    for vec_vvp in np.ndindex(mod_phi.shape):
        # print(vec_vvp, mvvp_phi[vec_vvp], mvvp_tau1[vec_vvp], mvvp_tau2[vec_vvp], sep='\n')

        # set simulation parameters and convert tau to inverse-tau for Gecko
        sim_params = {
            # TODO find a way to do this with sparse arrays
            'v_dc1':    mesh_V1[vec_vvp].item(),
            'v_dc2':    mesh_V2[vec_vvp].item(),
            'phi':      mod_phi[vec_vvp].item() / np.pi * 180,
            'tau1_inv': (np.pi - mod_tau1[vec_vvp].item()) / np.pi * 180,
            'tau2_inv': (np.pi - mod_tau2[vec_vvp].item()) / np.pi * 180
        }
        debug(sim_params)
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
            # values_mean = dab_converter.get_values(
            #     nodes=['p_dc1', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond'],
            #     operations=['mean']
            # )
            values_mean = dab_converter.get_values(
                nodes=l_means,
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
        debug("power_sim: %f / power_target: %f -> power_deviation: %f" % (values_mean['mean']['p_dc1'], mesh_P[vec_vvp].item(), power_deviation))
        # print("power_deviation", power_deviation)

        # save simulation results in array
        mvvp_iLs[vec_vvp] = values_rms['rms']['i_Ls']
        # ['p_dc1', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond']
        # mvvp_S11_p_sw[vec_vvp] = values_mean['mean']['S11_p_sw']
        for k, v in d_mvvp_means.items():
            v[vec_vvp] = values_mean['mean'][k]

    # TODO dirty hack to add the rms current
    d_mvvp_means['i_Ls'] = mvvp_iLs

    return d_mvvp_means


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module SIM ...")
