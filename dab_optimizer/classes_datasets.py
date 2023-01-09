#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###

import numpy as np
from dotmap import DotMap


class DAB_Specification(DotMap):
	"""
	Class to store the DAB specification.
	It contains only simple values of the same kind, e.g. float
	It inherits from DotMap to provide dot-notation usage instead of regular dict access.
	TODO limit input to e.g. float
	TODO define minimum dataset (keys and values that must exist)
	"""

	def save_to_array(self):
		return
		#todo
		#return spec_keys, spec_values

	def load_from_array(self, spec_keys, spec_values):
		return
		#todo


class DAB_Results(DotMap):
	"""
	Class to store simulation results.
	It contains only numpy arrays.
	It inherits from DotMap to provide dot-notation usage instead of regular dict access.
	TODO limit to np.ndarray
	TODO define minimum dataset (keys and values that must exist)
	"""



# class DAB_Specification:
# 	"""
# 	Class to store the DAB specification
# 	"""
# 	def __init__(self, V1_nom, V1_min, V1_max, V1_step, V2_nom, V2_min, V2_max, V2_step, P_min, P_max, P_nom, P_step,
# 				 n, L_s, L_m, fs_nom,
# 				 fs_min: float = None, fs_max: float = None, fs_step: float = None, L_c: float = None):
# 		self.V1_nom = V1_nom
# 		self.V1_min = V1_min
# 		self.V1_max = V1_max
# 		self.V1_step = V1_step
# 		self.V2_nom = V2_nom
# 		self.V2_min = V2_min
# 		self.V2_max = V2_max
# 		self.V2_step = V2_step
#
# 		self.P_nom = P_nom
# 		self.P_min = P_min
# 		self.P_max = P_max
# 		self.P_step = P_step
#
# 		self.n = n
# 		self.fs_nom = fs_nom
# 		self.fs_min = fs_min
# 		self.fs_max = fs_max
# 		self.fs_step = fs_step
#
# 		self.L_s = L_s
# 		self.L_m = L_m
# 		self.L_c = L_c
#
# 		# meshgrid for usage in e.g. matrix calculations or contour plot.
# 		# Link between array indices and x,y,z axes ranges.
# 		# sparse seems to be fine so far, if it troubles, then change it
# 		# sparse=False seems to be at least 2 times slower in following calculations!
# 		self.mesh_V1, self.mesh_V2, self.mesh_P = np.meshgrid(np.linspace(V1_min, V1_max, V1_step),
# 															  np.linspace(V2_min, V2_max, V2_step),
# 															  np.linspace(P_min, P_max, P_step), sparse=False)
#
#
# class DAB_Results:
# 	"""
# 	Class to store simulation results
# 	"""
#
# 	def __init__(self):
# 		self.specs= None
# 		self.mvvp_phi = None
# 		self.mvvp_tau1 = None
# 		self.mvvp_tau2 = None
# 		self.mvvp_iLs = None
# 		self.mvvp_S11_p_sw = None



# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of Module Datasets ...")

	# Set the basic DAB Specification
	# Setting it this way disables tab completion!
	dab_test_dict = {'V1_nom':700,
					 'V1_min': 600,
					 'V1_max': 800,
					 'V1_step': 3,
					 'V2_nom': 235,
					 'V2_min': 175,
					 'V2_max': 295,
					 'V2_step': 3,
					 'P_min': 400,
					 'P_max': 2200,
					 'P_nom': 2000,
					 'P_step': 3,
					 'n': 2.99,
					 'L_s': 84e-6,
					 'L_m': 599e-6,
					 'fs_nom': 200000
					 }
	dab_test_dm = DAB_Specification_DM(dab_test_dict)

	# Set the basic DAB Specification
	# Setting it this way enables tab completion!
	dab_test_dm = DAB_Specification_DM()
	dab_test_dm.V1_nom = 700
	dab_test_dm.V1_min = 600
	dab_test_dm.V1_max = 800
	dab_test_dm.V1_step = 3
	dab_test_dm.V2_nom = 235
	dab_test_dm.V2_min = 175
	dab_test_dm.V2_max = 295
	dab_test_dm.V2_step = 3
	dab_test_dm.P_min = 400
	dab_test_dm.P_max = 2200
	dab_test_dm.P_nom = 2000
	dab_test_dm.P_step = 3
	dab_test_dm.n = 2.99
	dab_test_dm.L_s = 84e-6
	dab_test_dm.L_m = 599e-6
	dab_test_dm.fs_nom = 200000

	# Some DotMap access examples
	print(dab_test_dm.V1_nom)
	print(dab_test_dm.V2_nom)
	print(dab_test_dm)
	print(dab_test_dm.toDict())
	dab_test_dm.pprint()
	dab_test_dm.pprint(pformat='json')

	# # OLD notation
	# dab_test = ds.DAB_Specification(V1_nom=700,
	# 								V1_min=600,
	# 								V1_max=800,
	# 								V1_step=3,
	# 								V2_nom=235,
	# 								V2_min=175,
	# 								V2_max=295,
	# 								V2_step=3,
	# 								P_min=400,
	# 								P_max=2200,
	# 								P_nom=2000,
	# 								P_step=3,
	# 								n=2.99,
	# 								L_s=84e-6,
	# 								L_m=599e-6,
	# 								fs_nom=200000,
	# 								)
	# print(dab_test.mesh_V1)
	# print(dab_test.mesh_V2)
	# print(dab_test.mesh_P)