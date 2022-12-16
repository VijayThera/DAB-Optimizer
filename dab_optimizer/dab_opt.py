#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###
import sys

import numpy as np
from collections import defaultdict
import math

import classes_datasets as ds
import debug_tools as db
import mod_cpm
import sim_gecko
import plot_dab




# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of DAB Optimizer ...")

	dab_test = ds.DAB_Specification(V1=700,
									V1_min=600,
									V1_max=800,
									V1_step=3,
									V2=235,
									V2_min=175,
									V2_max=295,
									V2_step=3,
									P_min=0,
									P_max=2200,
									P_nom=2000,
									P_step=3,
									n=2.99,
									L_s=84e-6,
									L_m=599e-6,
									fs=200000,
									)
	print(dab_test.mesh_V1)
	print(dab_test.mesh_V2)
	print(dab_test.mesh_P)
	# sys.exit(0)

	# using 3d dicts... ugly
	#d3d_phi, d3d_tau1, d3d_tau2 = mod_cpm.calc_modulation_dict(dab_test)

	# using np ndarray

	# Modulation Calculation
	mvvp_phi, mvvp_tau1, mvvp_tau2 = mod_cpm.calc_modulation(dab_test)

	# Simulation
	# mvvp_iLs, mvvp_S11_p_sw = sim_gecko.start_sim(dab_test, mvvp_phi, mvvp_tau1, mvvp_tau2)
	# print("mvvp_iLs: \n", mvvp_iLs)
	# print("mvvp_S11_p_sw: \n", mvvp_S11_p_sw)

	# Plotting
	plot_dab.plot_modulation(dab_test, mvvp_phi, mvvp_tau1, mvvp_tau2)

	# fake data
	U = np.exp(-(dab_test.mesh_V1/2) ** 2 - (dab_test.mesh_V2/3) ** 2 - dab_test.mesh_P ** 2)
	print(U)
	plot_dab.plot_rms_current(dab_test, U)

	plot_dab.show_plot()

