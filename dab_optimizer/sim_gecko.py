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
	simfilepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC.ipes'
	dab_converter = lpt.GeckoSimulation(simfilepath, timestep=50e-9, simtime=15e-6)

	params = dab_converter.get_global_parameters(['phi', 'tau1_inv', 'tau2_inv'])
	print(params)
	params = {'phi': 80.0, 'tau1_inv': 40.0, 'tau2_inv': 66.0}
	dab_converter.set_global_parameters(params)

	dab_converter.run_simulation()

	dab_converter.get_scope_data(node_names=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], file_name='test')

	values = dab_converter.get_values(nodes=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], operations=['mean', 'rms'], range_start_stop=[10e-6, 15e-6])
	print(values)




# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of Module SIM ...")