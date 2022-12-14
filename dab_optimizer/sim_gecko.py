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
def StartSim(DAB: ds.DAB_Specification, phi, tau1, tau2):
	print(DAB, phi, tau1, tau2)




# ---------- MAIN ----------
if __name__ == '__main__':
	print("Start of Module SIM ...")