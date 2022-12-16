#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###

import numpy as np
import math
from matplotlib import pyplot as plt

import classes_datasets as ds
import debug_tools as db


@db.timeit
def plot_modulation(DAB: ds.DAB_Specification, mvvp_phi, mvvp_tau1, mvvp_tau2):
	# plot
	fig, axs = plt.subplots(1, 3, sharey=True)
	fig.suptitle("DAB Modulation Angles")
	cf = axs[0].contourf(DAB.mesh_P[:,1,:], DAB.mesh_V2[:,1,:], mvvp_phi[:,1,:])
	axs[1].contourf(DAB.mesh_P[:,1,:], DAB.mesh_V2[:,1,:], mvvp_tau1[:,1,:])
	axs[2].contourf(DAB.mesh_P[:,1,:], DAB.mesh_V2[:,1,:], mvvp_tau2[:,1,:])
	axs[0].set_title("phi")
	axs[1].set_title("tau1")
	axs[2].set_title("tau2")
	for ax in axs.flat:
		ax.set(xlabel='P / W', ylabel='U2 / V')
		ax.label_outer()
	#fig.colorbar(cf, ax=axs.ravel().tolist())
	fig.colorbar(cf, ax=axs)

	plt.show()


