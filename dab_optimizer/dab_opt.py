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


def main():
	"""
	Run the complete optimization procedure
	"""
	# Set the basic DAB Specification
	# Setting it this way enables tab completion.
	dab_specs = ds.DAB_Specification()
	dab_specs.V1_nom = 700
	dab_specs.V1_min = 600
	dab_specs.V1_max = 800
	dab_specs.V1_step = 3
	dab_specs.V2_nom = 235
	dab_specs.V2_min = 175
	dab_specs.V2_max = 295
	dab_specs.V2_step = 3
	dab_specs.P_min = 400
	dab_specs.P_max = 2200
	dab_specs.P_nom = 2000
	dab_specs.P_step = 3
	dab_specs.n = 2.99
	dab_specs.L_s = 84e-6
	dab_specs.L_m = 599e-6
	dab_specs.fs_nom = 200000

	# Object to store all generated data
	dab_results = ds.DAB_Results()
	# gen mesh manually
	dab_results.mesh_V1, dab_results.mesh_V2, dab_results.mesh_P = np.meshgrid(
		np.linspace(dab_specs.V1_min, dab_specs.V1_max, dab_specs.V1_step),
		np.linspace(dab_specs.V2_min, dab_specs.V2_max, dab_specs.V2_step),
		np.linspace(dab_specs.P_min, dab_specs.P_max, dab_specs.P_step), sparse=False)

	# Modulation Calculation
	dab_results.mvvp_phi, dab_results.mvvp_tau1, dab_results.mvvp_tau2 = mod_cpm.calc_modulation(dab_specs.n,
																								 dab_specs.L_s,
																								 dab_specs.fs_nom,
																								 dab_results.mesh_V1,
																								 dab_results.mesh_V2,
																								 dab_results.mesh_P)

	# Simulation
	dab_results.mvvp_iLs, dab_results.mvvp_S11_p_sw = sim_gecko.start_sim(dab_results.mesh_V1,
																		  dab_results.mesh_V2,
																		  dab_results.mesh_P,
																		  dab_results.mvvp_phi,
																		  dab_results.mvvp_tau1,
																		  dab_results.mvvp_tau2)
	print("mvvp_iLs: \n", dab_results.mvvp_iLs)
	print("mvvp_S11_p_sw: \n", dab_results.mvvp_S11_p_sw)

	# Plotting
	pw = plotWindow()
	fig = plot_dab.plot_modulation(dab_results.mesh_V2,
								   dab_results.mesh_P,
								   dab_results.mvvp_phi,
								   dab_results.mvvp_tau1,
								   dab_results.mvvp_tau2)
	pw.addPlot("DAB Modulation Angles", fig)
	fig = plot_dab.plot_rms_current(dab_results.mesh_V2,
									dab_results.mesh_P,
									dab_results.mvvp_iLs)
	pw.addPlot("iLs", fig)
	fig = plot_dab.plot_rms_current(dab_results.mesh_V2,
									dab_results.mesh_P,
									dab_results.mvvp_S11_p_sw)
	pw.addPlot("S11 p_sw", fig)
	# plot_dab.show_plot()
	pw.show()


# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of DAB Optimizer ...")

	main()

# sys.exit(0)
