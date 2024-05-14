#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

import numpy as np
from collections import defaultdict
# For threads: run parallel in single process
# from threading import Thread, Lock
import threading as td
# For processes: run parallel on multiple cpu's
# from multiprocessing import Process, Lock
import multiprocessing as mp
# manage gecko java
# THIS makes jnius and Gecko stop working!!!
# import jnius
# Status bar
from tqdm import tqdm
# from time import sleep

import pygeckocircuits2 as pgc
from debug_tools import *


class Sim_Gecko:
    # mean values we want to get from the simulation
    l_means_keys = ['p_dc1', 'p_dc2', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond', 'S21_p_sw', 'S21_p_cond',
                    'S22_p_sw', 'S22_p_cond']
    l_rms_keys = ['i_Ls', 'i_Lm', 'v_dc1', 'i_dc1', 'v_dc2', 'i_dc2', 'i_C11', 'i_C12', 'i_C21', 'i_C22']
    mutex: td.Lock = None
    pbar: tqdm = None

    def __init__(self):
        # Number of threads or processes in parallel
        self.thread_count = 3
        # Init dict to store simulation result arrays
        self.da_sim_results = dict()
        # self.td_mutex = td.Lock()
        # self.mp_mutex = mp.Lock()

    @timeit
    def start_sim_threads(self, mesh_V1: np.ndarray, mesh_V2: np.ndarray,
                          mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                          simfilepath: str, timestep: float = None, simtime: float = None,
                          timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036,
                          gdebug: bool = False) -> dict:

        # Init arrays to store simulation results
        for k in self.l_means_keys:
            self.da_sim_results[k] = np.full_like(mod_phi, np.nan)
        for k in self.l_rms_keys:
            self.da_sim_results[k] = np.full_like(mod_phi, np.nan)

        # Progressbar init
        # Calc total number of iterations to simulate
        it_total = mod_phi.size
        self.pbar = tqdm(total=it_total)

        # ************ Gecko Start **********

        self.mutex = td.Lock()
        threads = []
        # Start the worker threads
        for i in range(self.thread_count):
            kwargs = {'mesh_V1':   mesh_V1, 'mesh_V2': mesh_V2, 'mod_phi': mod_phi, 'mod_tau1': mod_tau1,
                      'mod_tau2':  mod_tau2, 'simfilepath': simfilepath, 'timestep': timestep,
                      'simtime':   simtime, 'timestep_pre': timestep_pre, 'simtime_pre': simtime_pre,
                      'geckoport': geckoport + i, 'gdebug': gdebug}
            t = td.Thread(target=self._start_sim_single, kwargs=kwargs, name=str(i))
            t.start()
            threads.append(t)

        # Wait for the threads to complete
        for t in threads:
            t.join()

        # ************ Gecko End **********

        # Progressbar end
        self.pbar.close()

        # Rename the keys according to convention
        # da_sim_results_temp = dict()
        # for k, v in self.da_sim_results.items():
        #     da_sim_results_temp['sim_' + k] = v
        # self.da_sim_results = da_sim_results_temp

        debug(self.da_sim_results)
        return self.da_sim_results

    @timeit
    def start_sim_multi(self, mesh_V1: np.ndarray, mesh_V2: np.ndarray,
                        mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                        simfilepath: str, timestep: float = None, simtime: float = None,
                        timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036,
                        gdebug: bool = False) -> dict:

        # Init arrays to store simulation results
        for k in self.l_means_keys:
            self.da_sim_results[k] = np.full_like(mod_phi, np.nan)
        for k in self.l_rms_keys:
            self.da_sim_results[k] = np.full_like(mod_phi, np.nan)

        # Progressbar init
        # Calc total number of iterations to simulate
        it_total = mod_phi.size
        self.pbar = tqdm(total=it_total)

        # ************ Gecko Start **********

        self.mutex = mp.Lock()
        processes = []
        # Start the worker threads
        for i in range(self.thread_count):
            kwargs = {'mesh_V1':   mesh_V1, 'mesh_V2': mesh_V2, 'mod_phi': mod_phi, 'mod_tau1': mod_tau1,
                      'mod_tau2':  mod_tau2, 'simfilepath': simfilepath, 'timestep': timestep,
                      'simtime':   simtime, 'timestep_pre': timestep_pre, 'simtime_pre': simtime_pre,
                      'geckoport': geckoport + i, 'gdebug': gdebug}
            t = mp.Process(target=self._start_sim_single, kwargs=kwargs)
            t.start()
            processes.append(t)

        # Wait for the threads to complete
        for t in processes:
            t.join()

        # kwargs = {'mesh_V1':   mesh_V1, 'mesh_V2': mesh_V2, 'mod_phi': mod_phi, 'mod_tau1': mod_tau1,
        #           'mod_tau2':  mod_tau2, 'simfilepath': simfilepath, 'timestep': timestep,
        #           'simtime':   simtime, 'timestep_pre': timestep_pre, 'simtime_pre': simtime_pre,
        #           'geckoport': geckoport + 1, 'gdebug': gdebug}
        # self._start_sim_single(**kwargs)

        # self._start_sim_single(mesh_V1, mesh_V2, mod_phi, mod_tau1, mod_tau2, simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport)

        # ************ Gecko End **********

        # Progressbar end
        self.pbar.close()

        # Rename the keys according to convention
        # da_sim_results_temp = dict()
        # for k, v in self.da_sim_results.items():
        #     da_sim_results_temp['sim_' + k] = v
        # self.da_sim_results = da_sim_results_temp

        debug(self.da_sim_results)
        return self.da_sim_results

    def _start_sim_single(self, mesh_V1: np.ndarray, mesh_V2: np.ndarray,
                          mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                          simfilepath: str, timestep: float = None, simtime: float = None,
                          timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036,
                          gdebug: bool = False):

        info(geckoport, simfilepath)

        # ************ Gecko Start **********

        try:
            # use pyjnius here

            if not __debug__:
                # Gecko Basics
                dab_converter = pgc.GeckoSimulation(simfilepath=simfilepath, geckoport=geckoport, debug=gdebug)

            for vec_vvp in np.ndindex(mod_phi.shape):
                # debug(vec_vvp, mod_phi[vec_vvp], mod_tau1[vec_vvp], mod_tau2[vec_vvp], sep='\n')

                # set simulation parameters and convert tau to inverse-tau for Gecko
                sim_params = {
                    # TODO find a way to do this with sparse arrays
                    'v_dc1': mesh_V1[vec_vvp].item(),
                    'v_dc2': mesh_V2[vec_vvp].item(),
                    'phi':   mod_phi[vec_vvp].item() / np.pi * 180,
                    'tau1':  mod_tau1[vec_vvp].item() / np.pi * 180,
                    'tau2':  mod_tau2[vec_vvp].item() / np.pi * 180
                    # Old v1 Model needed inverse tau
                    # 'tau1_inv': (np.pi - mod_tau1[vec_vvp].item()) / np.pi * 180,
                    # 'tau2_inv': (np.pi - mod_tau2[vec_vvp].item()) / np.pi * 180
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
                        nodes=self.l_means_keys,
                        operations=['mean']
                    )
                    values_rms = dab_converter.get_values(
                        nodes=self.l_rms_keys,
                        operations=['rms']
                    )
                else:
                    # generate some fake data for debugging
                    # values_mean = {'mean': {'p_dc1':      np.random.uniform(0.0, 1000),
                    #                         'S11_p_sw':   np.random.uniform(0.0, 10),
                    #                         'S11_p_cond': np.random.uniform(0.0, 10),
                    #                         'S12_p_sw':   np.random.uniform(0.0, 1000),
                    #                         'S12_p_cond': np.random.uniform(0.0, 100)}}
                    # values_rms = {'rms': {'i_Ls': np.random.uniform(0.0, 10)}}
                    values_mean = defaultdict(dict)
                    values_rms = defaultdict(dict)
                    for k in self.l_means_keys:
                        values_mean['mean'][k] = np.random.uniform(0.0, 1)
                    for k in self.l_rms_keys:
                        values_rms['rms'][k] = np.random.uniform(0.0, 1)

                # ***** LOCK Start *****
                self.mutex.acquire()
                # save simulation results in arrays
                for k in self.l_means_keys:
                    self.da_sim_results[k][vec_vvp] = values_mean['mean'][k]
                for k in self.l_rms_keys:
                    self.da_sim_results[k][vec_vvp] = values_rms['rms'][k]

                # Progressbar update, default increment +1
                self.pbar.update()
                self.mutex.release()
                # ***** LOCK End *****

            # dab_converter.__del__()

        finally:
            # jnius.detach()
            pass
        # ************ Gecko End **********


@timeit
def start_sim(mesh_V1: np.ndarray, mesh_V2: np.ndarray,
              mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
              simfilepath: str, timestep: float = None, simtime: float = None,
              timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036, gdebug: bool = False) -> dict:
    # mean values we want to get from the simulation
    l_means_keys = ['p_dc1', 'p_dc2', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond', 'S21_p_sw', 'S21_p_cond',
                    'S22_p_sw', 'S22_p_cond']
    l_rms_keys = ['i_Ls', 'i_Lm', 'v_dc1', 'i_dc1', 'v_dc2', 'i_dc2', 'i_C11', 'i_C12', 'i_C21', 'i_C22']

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
        dab_converter = pgc.GeckoSimulation(simfilepath=simfilepath, geckoport=geckoport, debug=gdebug)

    for vec_vvp in np.ndindex(mod_phi.shape):
        # debug(vec_vvp, mod_phi[vec_vvp], mod_tau1[vec_vvp], mod_tau2[vec_vvp], sep='\n')

        # set simulation parameters and convert tau to inverse-tau for Gecko
        sim_params = {
            # TODO find a way to do this with sparse arrays
            'v_dc1': mesh_V1[vec_vvp].item(),
            'v_dc2': mesh_V2[vec_vvp].item(),
            'phi':   mod_phi[vec_vvp].item() / np.pi * 180,
            'tau1':  mod_tau1[vec_vvp].item() / np.pi * 180,
            'tau2':  mod_tau2[vec_vvp].item() / np.pi * 180
            # Old v1 Model needed inverse tau
            # 'tau1_inv': (np.pi - mod_tau1[vec_vvp].item()) / np.pi * 180,
            # 'tau2_inv': (np.pi - mod_tau2[vec_vvp].item()) / np.pi * 180
        }
        # info(sim_params)

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
            # values_mean = {'mean': {'p_dc1':      np.random.uniform(0.0, 1000),
            #                         'S11_p_sw':   np.random.uniform(0.0, 1),
            #                         'S11_p_cond': np.random.uniform(0.0, 10),
            #                         'S12_p_sw':   np.random.uniform(0.0, 1),
            #                         'S12_p_cond': np.random.uniform(0.0, 10)}}
            # values_rms = {'rms': {'i_Ls': np.random.uniform(0.0, 10)}}
            values_mean = defaultdict(dict)
            values_rms = defaultdict(dict)
            for k in l_means_keys:
                values_mean['mean'][k] = np.random.uniform(0.0, 1)
            for k in l_rms_keys:
                values_rms['rms'][k] = np.random.uniform(0.0, 1)

        # save simulation results in arrays
        for k in l_means_keys:
            da_sim_results[k][vec_vvp] = values_mean['mean'][k]
        for k in l_rms_keys:
            da_sim_results[k][vec_vvp] = values_rms['rms'][k]
        # info(values_mean)

        # Progressbar update, default increment +1
        pbar.update()

    if not __debug__:
        # Gecko Basics
        dab_converter.__del__()
    # ************ Gecko End **********

    # Progressbar end
    pbar.close()

    # Rename the keys according to convention
    # da_sim_results_temp = dict()
    # for k, v in da_sim_results.items():
    #     da_sim_results_temp['sim_' + k] = v
    # da_sim_results = da_sim_results_temp

    # info(da_sim_results)
    return da_sim_results


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module SIM ...")
