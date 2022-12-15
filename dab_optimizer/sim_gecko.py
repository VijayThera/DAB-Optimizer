#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###

import os
import pathlib
import sys
import itertools
import numpy as np
from collections import defaultdict
import math

import leapythontoolbox as lpt
import classes_datasets as ds
import debug_tools as db



@db.timeit
def StartSim(DAB: ds.DAB_Specification, phi, tau1, tau2):
	sim_filepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC.ipes'
	dab_converter = lpt.GeckoSimulation(sim_filepath)

	mvvp_iLs = np.zeros(shape=(1,2,3))

	#TODO ugly but for testing until np.array is implemented
	for V1, V2, P in itertools.product(range(DAB.V1_min, DAB.V1_max + 1, 100),
									   range(DAB.V2_min, DAB.V2_max + 1, 60),
									   range(DAB.P_min, DAB.P_max + 1, 200)):
		print(V1, V2, P, phi[V1][V2][P], tau1[V1][V2][P], tau2[V1][V2][P])
		sim_params = {
			'v_dc1': V1,
			'v_dc2': V2,
			'phi': phi[V1][V2][P],
			'tau1_inv': (math.pi - tau1[V1][V2][P]) / math.pi * 180,
			'tau2_inv': (math.pi - tau2[V1][V2][P]) / math.pi * 180
		}
		dab_converter.set_global_parameters(sim_params)
		#TODO time settings should be variable
		dab_converter.run_simulation(timestep=100e-12, simtime=15e-6, timestep_pre=50e-9, simtime_pre=10e-3)
		values_mean = dab_converter.get_values(
			nodes=['p_dc1', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond'],
			operations=['mean']
		)
		values_rms = dab_converter.get_values(
			nodes=['i_Ls'],
			operations=['rms']
		)
		power_deviation = values_mean['mean']['p_dc1'] / P





	# params = dab_converter.get_global_parameters(['phi', 'tau1_inv', 'tau2_inv'])
	# print(params)
	# params = {'phi': 80.0, 'tau1_inv': 40.0, 'tau2_inv': 66.0}
	# dab_converter.set_global_parameters(params)
	#
	# dab_converter.run_simulation(timestep=50e-9, simtime=15e-6)
	#
	# dab_converter.get_scope_data(node_names=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], file_name='test')
	#
	# values = dab_converter.get_values(nodes=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], operations=['mean', 'rms'], range_start_stop=[10e-6, 15e-6])
	# print(values)




# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of Module SIM ...")