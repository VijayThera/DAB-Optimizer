#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###
import sys

import pickle
import numpy as np
from collections import defaultdict
import math

import classes_datasets as ds
import debug_tools as db
import mod_cpm
import sim_gecko
import plot_dab

from plotWindow import plotWindow




# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of DAB Optimizer ...")

	dab_test = ds.DAB_Specification(V1_nom=700,
									V1_min=600,
									V1_max=800,
									V1_step=3,
									V2_nom=235,
									V2_min=175,
									V2_max=295,
									V2_step=3,
									P_min=400,
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

	# Object to store data
	dab_results = ds.DAB_Results
	dab_results.specs = dab_test

	# using 3d dicts... ugly
	#d3d_phi, d3d_tau1, d3d_tau2 = mod_cpm.calc_modulation_dict(dab_test)

	# using np ndarray

	# Modulation Calculation
	mvvp_phi, mvvp_tau1, mvvp_tau2 = mod_cpm.calc_modulation(dab_test)
	dab_results.mvvp_phi = mvvp_phi
	dab_results.mvvp_tau1 = mvvp_tau1
	dab_results.mvvp_tau2 = mvvp_tau2

	# Simulation
	mvvp_iLs, mvvp_S11_p_sw = sim_gecko.start_sim(dab_test, mvvp_phi, mvvp_tau1, mvvp_tau2)
	print("mvvp_iLs: \n", mvvp_iLs)
	print("mvvp_S11_p_sw: \n", mvvp_S11_p_sw)
	dab_results.mvvp_iLs = mvvp_iLs
	dab_results.mvvp_S11_p_sw = mvvp_S11_p_sw

	# Save results to file
	with open('dab_results.pickle', 'wb') as f:
		pickle.dump(dab_results, f)

	# Load results from file
	with open('dab_results.pickle') as f:
		dab_results = pickle.load(f)

	# Plotting
	pw = plotWindow()

	fig = plot_dab.plot_modulation(dab_results.specs, dab_results.mvvp_phi, dab_results.mvvp_tau1, dab_results.mvvp_tau2)
	pw.addPlot("DAB Modulation Angles", fig)

	fig = plot_dab.plot_rms_current(dab_results.specs, dab_results.mvvp_iLs)
	pw.addPlot("iLs", fig)

	fig = plot_dab.plot_rms_current(dab_results.specs, dab_results.mvvp_S11_p_sw)
	pw.addPlot("S11 p_sw", fig)

	#plot_dab.show_plot()
	pw.show()

