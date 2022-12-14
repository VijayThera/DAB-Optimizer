#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###

import numpy as np
from collections import defaultdict
import math

import classes_datasets as ds
import debug_tools as db
import mod_cpm
import sim_gecko




# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of DAB Optimizer ...")

	dab_test = ds.DAB_Specification(V1=700,
									V1_min=600,
									V1_max=800,
									V2=235,
									V2_min=175,
									V2_max=295,
									P_min=0,
									P_max=2200,
									P_nom=2000,
									n=2.99,
									L_s=84e-6,
									L_m=599e-6,
									fs=200000,
									)

	d3d_phi, d3d_tau1, d3d_tau2 = mod_cpm.CalcModulation(dab_test)
	sim_gecko.StartSim(dab_test, d3d_phi, d3d_tau1, d3d_tau2)

