#!/usr/bin/python3
# coding: utf-8
# python >= 3.10

import numpy as np
# for threads
# import threading
# run parallel on multiple cpu's
import multiprocessing as mp
# manage gecko java
import jnius
# Status bar
from time import sleep
from tqdm import tqdm

import leapythontoolbox as lpt
import classes_datasets as ds
from debug_tools import *


class Sim_Gecko:

    @timeit
    def start_sim_multi(self, mesh_V1: np.ndarray, mesh_V2: np.ndarray, mesh_P: np.ndarray,
                           mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                           num_threads: int = 6) -> dict:

        try:
            # use pyjnius here
            True
        finally:
            jnius.detach()

    @timeit
    def _start_sim_single(self, mesh_V1: np.ndarray, mesh_V2: np.ndarray, mesh_P: np.ndarray,
                          mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray):
        True


@timeit
def start_sim(mesh_V1: np.ndarray, mesh_V2: np.ndarray,
              mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
              simfilepath: str, timestep: float = None, simtime: float = None,
              timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036, debug: bool = False) -> dict:
    # mean values we want to get from the simulation
    l_means_keys = ['p_dc1', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond']
    l_rms_keys = ['i_Ls']

    # Init arrays to store simulation results
    da_sim_results = dict()
    for k in l_means_keys:
        da_sim_results[k] = np.full_like(mod_phi, np.nan)
    for k in l_rms_keys:
        da_sim_results[k] = np.full_like(mod_phi, np.nan)

    # Progressbar init
    # Calc total number of iterations to simulate
    it_total = mod_phi.size
    pbar = tqdm(total=it_total)

    # ************ Gecko Start **********
    if not __debug__:
        # Gecko Basics
        dab_converter = lpt.GeckoSimulation(simfilepath=simfilepath, geckoport=geckoport, debug=debug)

    for vec_vvp in np.ndindex(mod_phi.shape):
        # debug(vec_vvp, mod_phi[vec_vvp], mod_tau1[vec_vvp], mod_tau2[vec_vvp], sep='\n')

        # set simulation parameters and convert tau to inverse-tau for Gecko
        sim_params = {
            # TODO find a way to do this with sparse arrays
            'v_dc1':    mesh_V1[vec_vvp].item(),
            'v_dc2':    mesh_V2[vec_vvp].item(),
            'phi':      mod_phi[vec_vvp].item() / np.pi * 180,
            'tau1_inv': (np.pi - mod_tau1[vec_vvp].item()) / np.pi * 180,
            'tau2_inv': (np.pi - mod_tau2[vec_vvp].item()) / np.pi * 180
        }
        # debug(sim_params)

        # start simulation for this operation point
        # TODO optimize for multithreading, maybe multiple Gecko instances needed
        if not __debug__:
            dab_converter.set_global_parameters(sim_params)
        if not __debug__:
            # TODO time settings should be variable
            # dab_converter.run_simulation(timestep=100e-12, simtime=15e-6, timestep_pre=50e-9, simtime_pre=10e-3)
            # TODO Bug in LPT with _pre settings! Does this still run a pre-simulation like in the model?
            # Start the simulation and get the results
            dab_converter.run_simulation(timestep=timestep, simtime=simtime)
            values_mean = dab_converter.get_values(
                nodes=l_means_keys,
                operations=['mean']
            )
            values_rms = dab_converter.get_values(
                nodes=l_rms_keys,
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

        # save simulation results in arrays
        for k in l_means_keys:
            da_sim_results[k][vec_vvp] = values_mean['mean'][k]
        for k in l_rms_keys:
            da_sim_results[k][vec_vvp] = values_rms['rms'][k]

        # Progressbar update, default increment +1
        pbar.update()
    # ************ Gecko End **********

    # Progressbar end
    pbar.close()
    # debug(da_sim_results)
    return da_sim_results


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module SIM ...")
