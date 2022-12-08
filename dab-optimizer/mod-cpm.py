#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###
import itertools

import numpy as np
from collections import defaultdict
import math
import classes_datasets



def CalcModulation(DAB: classes_datasets.DAB_Specification):
	step = 1
	d3d_phi = defaultdict(lambda : defaultdict(dict))
	for V1, V2, P in itertools.product(range(DAB.V1_min, DAB.V1_max, step),
									   range(DAB.V2_min, DAB.V2_max, step),
									   range(DAB.P_min, DAB.P_max, step)):
		d3d_phi[V1][V2][P] = calc_phi(V1, V2, P, DAB.fs, DAB.L_s, DAB.n)

	print(d3d_phi)


def calc_phi(V1, V2, P, fs, L, n):
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

	dab_test = classes_datasets.DAB_Specification(V1=700,
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
	CalcModulation(dab_test)