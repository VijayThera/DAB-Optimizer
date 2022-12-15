#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###

import itertools
import numpy as np
from collections import defaultdict
import math

import classes_datasets as ds
import debug_tools as db


@db.timeit
def calc_modulation(DAB: ds.DAB_Specification):
	# init 3d arrays
	mvvp_phi = np.zeros_like(DAB.mesh_V1, dtype=float)
	# init these with pi because they are constant for CPM
	mvvp_tau1 = np.full_like(DAB.mesh_V1, np.pi, dtype=float)
	mvvp_tau2 = np.full_like(DAB.mesh_V1, np.pi, dtype=float)

	# Calculate phase shift difference from input to output bridge
	#TODO maybe have to consider the case sqrt(<0). When does this happen?
	mvvp_phi = np.pi / 2 * (1 - np.sqrt(1 - (8 * DAB.fs * DAB.L_s * DAB.mesh_P) / (DAB.n * DAB.mesh_V1 * DAB.mesh_V2)))

	return mvvp_phi, mvvp_tau1, mvvp_tau2

@db.timeit
def calc_modulation_dict(DAB: ds.DAB_Specification):
	#step = 10
	d3d_phi = defaultdict(lambda: defaultdict(dict))
	d3d_tau1 = defaultdict(lambda: defaultdict(dict))
	d3d_tau2 = defaultdict(lambda: defaultdict(dict))
	# ugly but for testing until np.array is implemented
	for V1, V2, P in itertools.product(range(DAB.V1_min, DAB.V1_max+1, 100),
									   range(DAB.V2_min, DAB.V2_max+1, 60),
									   range(DAB.P_min, DAB.P_max+1, 200)):
		d3d_phi[V1][V2][P] = _calc_phi(V1, V2, P, DAB.fs, DAB.L_s, DAB.n)
		d3d_tau1[V1][V2][P] = math.pi
		d3d_tau2[V1][V2][P] = math.pi

	return d3d_phi, d3d_tau1, d3d_tau2


def _calc_phi(V1, V2, P, fs, L, n):
	phi = math.pi / 2 * (1 - math.sqrt(1 - (8 * fs * L * P) / (n * V1 * V2)))
	return phi


def f_mln_phi(mln_lambda: np.array, mln_n: np.array, v1: float, v2: float, power_max: float) -> np.array:
	"""
	Calculate phase shift difference from input to output bridge
	:param mln_lambda: lambda in ln-mesh
	:type mln_lambda: np.array
	:param mln_n: n in ln-mesh
	:type mln_n: np.array
	:param v1: input voltage
	:type v1: float
	:param v2: output voltage   git dif
	:type v2: float
	:param power_max: power
	:type power_max: float
	:return: phi (phase shift difference from input to output bridge)
	:rtype: np.array
	"""
	mln_phi = 1 - (8 * mln_lambda * power_max) / (mln_n * v1 * v2)

	mln_phi[mln_phi < 0] = np.nan
	mln_phi[~np.isnan(mln_phi)] = np.pi / 2 * (1 - np.sqrt(mln_phi[~np.isnan(mln_phi)]))

	return mln_phi


# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of Module CPM ...")

	dab_test = ds.DAB_Specification(V1=700,
									V1_min=600,
									V1_max=800,
									V1_step=30,
									V2=235,
									V2_min=175,
									V2_max=295,
									V2_step=30,
									P_min=0,
									P_max=2200,
									P_nom=2000,
									P_step=30,
									n=2.99,
									L_s=84e-6,
									L_m=599e-6,
									fs=200000,
									)

	# using 3d dicts... ugly
	d3d_phi, d3d_tau1, d3d_tau2 = calc_modulation_dict(dab_test)
	print(d3d_phi)

	# using np ndarray
	mvvp_phi, mvvp_tau1, mvvp_tau2 = calc_modulation(dab_test)
	print(mvvp_phi)

	print(dab_test.mesh_V1, dab_test.mesh_V2, dab_test.mesh_P, sep='\n')
