#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

import numpy as np
from dotmap import DotMap

from debug_tools import *


class DAB_Specification(DotMap):
    """
    Class to store the DAB specification.
    It contains only simple values of the same kind, e.g. float
    It inherits from DotMap to provide dot-notation usage instead of regular dict access.
    TODO define minimum dataset (keys and values that must exist)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialisation with an other Dict is not handled and type converted yet!
        :param args:
        :param kwargs:
        """
        if args or kwargs:
            warning("Don't use this type of initialisation!")
        # if kwargs:
        #     d.update((k, float(v)) for k,v in self.__call_items(kwargs)
        super().__init__(*args, **kwargs)

    def __setitem__(self, k, v):
        # Convert all values to float
        super().__setitem__(k, float(v))

    # def __setattr__(self, k, v):
    #     print(f'Setting {k} to {v}')
    #     print(f'Type {type(k)} to {type(v)}')
    #     super().__setattr__(k, v)

    def export_to_array(self):
        """
        Exports the items of the dict to separate np.arrays.
        spec_keys containing the dict keys as strings.
        spec_values containing the dict values as float.
        The order of the elements in the array must be kept!
        :return: spec_keys, spec_values
        """
        spec_keys = np.array(list(self.keys()))
        spec_values = np.array(list(self.values()))
        return spec_keys, spec_values

    def import_from_array(self, spec_keys, spec_values):
        """
        Import a set of array from a previous export.
        Both numpy arrays must be 1D, of the same length and in order.
        :param spec_keys: np.array containing the keys (strings) for the dict.
        :param spec_values: np.array containing the (float) values for the dict.
        """
        if (len(spec_keys) == len(spec_values)) and (spec_keys.dtype.type is np.str_) and (
                spec_values.dtype.type is np.float_):
            for i in range(len(spec_keys)):
                self[spec_keys.item(i)] = spec_values.item(i)
        else:
            warning("Arrays are not valid for import!")


class DAB_Results(DotMap):
    """
    Class to store simulation results.
    It contains only numpy arrays, you can only add those.
    It inherits from DotMap to provide dot-notation usage instead of regular dict access.
    Make sure your key names start with one of the "_allowed_keys", if not you can not add the key.
    Add a useful name string after the prefix from "_allowed_keys" to identify your results later.
    """

    _allowed_keys = ['_timestamp', '_comment', 'mesh_', 'mod_', 'sim_', 'coss_', 'qoss_']

    def __init__(self, *args, **kwargs):
        """
        Initialisation with an other Dict is not handled and type converted yet!
        :param args:
        :param kwargs:
        """
        if args or kwargs:
            warning("Don't use this type of initialisation!")
        # if kwargs:
        #     d.update((k, float(v)) for k,v in self.__call_items(kwargs)
        super().__init__(*args, **kwargs)

    def __setitem__(self, k, v):
        # Only np.ndarray is allowed
        if isinstance(v, np.ndarray):
            # Check for allowed key names
            if any(k.startswith(allowed_key) for allowed_key in self._allowed_keys):
                super().__setitem__(k, v)
            else:
                warning('None of the _allowed_keys are used! Nothing added! Used key: ' + str(k))
        else:
            warning('Value is not an numpy ndarray! Nothing added! Used type: ' + str(type(v)))

    def gen_meshes(self, V1_min: float, V1_max: float, V1_step,
                        V2_min: float, V2_max: float, V2_step,
                        P_min: float, P_max: float, P_step):
        """
        Generates the default meshgrids for V1, V2 and P.
        Set the min and max values for your meshgrid with a step size (linspace) of step value.
        :param V1_min:
        :param V1_max:
        :param V1_step: int or float (converted to int) are accepted
        :param V2_min:
        :param V2_max:
        :param V2_step: int or float (converted to int) are accepted
        :param P_min:
        :param P_max:
        :param P_step: int or float (converted to int) are accepted
        """
        self.mesh_V1, self.mesh_V2, self.mesh_P = np.meshgrid(np.linspace(V1_min, V1_max, int(V1_step)),
                                                              np.linspace(V2_min, V2_max, int(V2_step)),
                                                              np.linspace(P_min, P_max, int(P_step)), sparse=False)

    def append_result_dict(self, result: dict, name_pre: str = '', name_post: str = ''):
        # Unpack the results
        for k, v in result.items():
            self[name_pre + k + name_post] = v


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
    # Don't use this!
    dab_test_dict = {'V1_nom':  700,
                     'V1_min':  600,
                     'V1_max':  800,
                     'V1_step': 3,
                     'V2_nom':  235,
                     'V2_min':  175,
                     'V2_max':  295,
                     'V2_step': 3,
                     'P_min':   400,
                     'P_max':   2200,
                     'P_nom':   2000,
                     'P_step':  3,
                     'n':       2.99,
                     'L_s':     84e-6,
                     'L_m':     599e-6,
                     'fs_nom':  200000
                     }
    dab_test_dm_no_completion = DAB_Specification(dab_test_dict)
    # Check Value types
    for value in dab_test_dm_no_completion.values():
        print(type(value))

    # Set the basic DAB Specification
    # Setting it this way enables tab completion!
    dab_test_dm = DAB_Specification()
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
    print(dab_test_dm['V2_nom'])
    print(dab_test_dm)
    print(dab_test_dm.toDict())
    dab_test_dm.pprint()
    dab_test_dm.pprint(pformat='json')
    # Check Value types
    for value in dab_test_dm.values():
        print(type(value))

    # export
    print("export")
    spec_keys, spec_values = dab_test_dm.export_to_array()
    print(spec_keys, spec_values)
    # import
    print("import")
    dab_loaded = DAB_Specification()
    dab_loaded.import_from_array(spec_keys, spec_values)
    print(dab_loaded)

    # a = np.fromiter(dab_test_dm.items(), dtype=dtype, count=len(dab_test_dm))

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
