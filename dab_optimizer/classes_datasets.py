#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###

import numpy as np
from dotmap import DotMap


class DAB_Specification_DM(DotMap):
	"""
	Class to store the DAB specification
	"""




class DAB_Specification:
	"""
	Class to store the DAB specification
	"""
	def __init__(self, V1_nom, V1_min, V1_max, V1_step, V2_nom, V2_min, V2_max, V2_step, P_min, P_max, P_nom, P_step,
				 n, L_s, L_m, fs,
				 fs_min: float = None, fs_max: float = None, L_c: float = None):
		self.V1_nom = V1_nom
		self.V1_min = V1_min
		self.V1_max = V1_max
		self.V1_step = V1_step
		self.V2_nom = V2_nom
		self.V2_min = V2_min
		self.V2_max = V2_max
		self.V2_step = V2_step

		self.P_nom = P_nom
		self.P_min = P_min
		self.P_max = P_max
		self.P_step = P_step

		self.n = n
		self.fs = fs
		self.fs_min = fs_min
		self.fs_max = fs_max

		self.L_s = L_s
		self.L_m = L_m
		self.L_c = L_c

		# meshgrid for usage in e.g. matrix calculations or contour plot.
		# Link between array indices and x,y,z axes ranges.
		# sparse seems to be fine so far, if it troubles, then change it
		# sparse=False seems to be at least 2 times slower in following calculations!
		self.mesh_V1, self.mesh_V2, self.mesh_P = np.meshgrid(np.linspace(V1_min, V1_max, V1_step),
															  np.linspace(V2_min, V2_max, V2_step),
															  np.linspace(P_min, P_max, P_step), sparse=False)


class DAB_Results:
	"""
	Class to store simulation results
	"""

	def __init__(self):
		self.specs= None
		self.mvvp_phi = None
		self.mvvp_tau1 = None
		self.mvvp_tau2 = None
		self.mvvp_iLs = None
		self.mvvp_S11_p_sw = None