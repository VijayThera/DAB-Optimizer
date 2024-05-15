#!/usr/bin/python3
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# coding: utf-8
# python >= 3.10

"""
        DAB Modulation Toolbox
        Copyright (C) 2023  strayedelectron

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU Affero General Public License as
        published by the Free Software Foundation, either version 3 of the
        License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU Affero General Public License for more details.

        You should have received a copy of the GNU Affero General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import pprint
import numpy as np
from dotmap import DotMap

from debug_tools import *


class DAB_Data(DotMap):
    """
    Class to store the DAB specification, modulation and simulation results and some more data.
    It contains only numpy arrays, you can only add those.
    In fact everything is a numpy array even single int or float values!
    It inherits from DotMap to provide dot-notation usage instead of regular dict access.
    Make sure your key names start with one of the "_allowed_keys", if not you can not add the key.
    Add a useful name string after the prefix from "_allowed_keys" to identify your results later.
    """

    _allowed_keys = ['_timestamp', '_comment', 'spec_', 'mesh_', 'mod_', 'sim_', 'meas_', 'coss_', 'qoss_', 'iter_']
    _allowed_spec_keys = ['V1_nom', 'V1_min', 'V1_max', 'V1_step', 'V2_nom', 'V2_min', 'V2_max', 'V2_step', 'P_min',
                          'P_max', 'P_nom', 'P_step', 'n', 'Ls', 'Lm', 'Lc1', 'Lc2', 'Lc2_', 'fs', 't_dead1', 't_dead2',
                          'temp', 'C_HB11', 'C_HB12', 'C_HB21', 'C_HB22',
                          # deprecated:
                          'C_PCB_leg', 'L_s', 'L_m', 'fs_nom']

    def __init__(self, *args, **kwargs):
        """
        Initialisation with another Dict is not handled and type converted yet!
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
            if any(k.startswith(allowed_key) for allowed_key in (self._allowed_keys + self._allowed_spec_keys)):
                super().__setitem__(k, v)
            else:
                warning('None of the _allowed_keys are used! Nothing added! Used key: ' + str(k))
        else:
            # Value will be converted to a ndarray
            # Check for allowed key names
            if any(k.startswith(allowed_key) for allowed_key in (self._allowed_keys + self._allowed_spec_keys)):
                super().__setitem__(k, np.asarray(v))
            else:
                warning('None of the _allowed_keys are used! Nothing added! Used key: ' + str(k))

    def pprint_to_file(self, filename):
        """
        Prints the DAB in nice human-readable form into a text file.
        WARNING: This file can not be loaded again! It is only for documentation.
        :param filename:
        """
        filename = os.path.expanduser(filename)
        filename = os.path.expandvars(filename)
        filename = os.path.abspath(filename)
        if os.path.isfile(filename):
            warning("File already exists!")
        else:
            with open(filename, 'w') as file:
                pprint.pprint(self.toDict(), file)

    def save_to_file(self, directory=str(), name=str(), timestamp=True, comment=str()):
        """
        Save everything (except plots) in one file.
        WARNING: Existing files will be overwritten!

        File is ZIP compressed and contains several named np.ndarray objects:
            # String is constructed as follows:
            # used module (e.g. "mod_sps_") + value name (e.g. "phi")
            mod_sps_phi: mod_sps calculated values for phi
            mod_sps_tau1: mod_sps calculated values for tau1
            mod_sps_tau2: mod_sps calculated values for tau1
            sim_sps_iLs: simulation results with mod_sps for iLs
            sim_sps_S11_p_sw:

        :param directory: Folder where to save the files
        :param name: String added to the filename. Without file extension. Datetime may prepend the final name.
        :param timestamp: If the datetime should prepend the final name. default True
        :param comment:
        """

        # Add some descriptive data to the file
        # Adding a timestamp, it may be useful
        self['_timestamp'] = np.asarray(datetime.now().isoformat())
        # Adding a comment to the file, hopefully a descriptive one
        if comment:
            self['_comment'] = np.asarray(comment)

        # Adding a timestamp to the filename if requested
        if timestamp:
            if name:
                filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
            else:
                filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        else:
            if name:
                filename = name
            else:
                # set some default non-empty filename
                filename = "dab_dataset"

        if directory:
            directory = os.path.expanduser(directory)
            directory = os.path.expandvars(directory)
            directory = os.path.abspath(directory)
            if os.path.isdir(directory):
                file = os.path.join(directory, filename)
            else:
                warning("Directory does not exist!")
                file = os.path.join(filename)
        else:
            file = os.path.join(filename)

        # numpy saves everything for us in a handy zip file
        np.savez_compressed(file=file, **self)

    def gen_meshes(self):
        """
        Generates the default meshgrids for V1, V2 and P.
        Values for:
        'V1_nom', 'V1_min', 'V1_max', 'V1_step',
        'V2_nom', 'V2_min', 'V2_max', 'V2_step',
        'P_min', 'P_max', 'P_nom', 'P_step'
        must be set first!
        """
        self.mesh_V1, self.mesh_V2, self.mesh_P = np.meshgrid(np.linspace(self.V1_min, self.V1_max, int(self.V1_step)),
                                                              np.linspace(self.V2_min, self.V2_max, int(self.V2_step)),
                                                              np.linspace(self.P_min, self.P_max, int(self.P_step)),
                                                              sparse=False)

    def append_result_dict(self, result: dict, name_pre: str = '', name_post: str = ''):
        # Unpack the results
        for k, v in result.items():
            self[name_pre + k + name_post] = v

    def import_Coss(self, file: str(), name: str()):
        """
        Import a csv file containing the Coss(Vds) capacitance from the MOSFET datasheet.
        This may be generated with: https://apps.automeris.io/wpd/

        Note we assume V_ds in Volt and C_oss in F. If this is not the case scale your data accordingly!

        CSV File should look like this:
        # V_ds / V; C_oss / F
        1,00; 900,00e-12
        2,00; 800,00e-12
        :param file: csv file path
        """
        file = os.path.expanduser(file)
        file = os.path.expandvars(file)
        file = os.path.abspath(file)

        # Conversion from decimal separator comma to point so that np can read floats
        # Be careful if your csv is actually comma separated! ;)
        def conv(x):
            return x.replace(',', '.').encode()

        # Read csv file
        csv_data = np.genfromtxt((conv(x) for x in open(file)), delimiter=';', dtype=float)

        # Maybe check if data is monotonically
        # Check if voltage is monotonically rising
        if not np.all(csv_data[1:, 0] >= csv_data[:-1, 0], axis=0):
            warning("The voltage in csv file is not monotonically rising!")
        # Check if Coss is monotonically falling
        if not np.all(csv_data[1:, 1] <= csv_data[:-1, 1], axis=0):
            warning("The C_oss in csv file is not monotonically falling!")

        # Rescale and interpolate the csv data to have a nice 1V step size from 0V to v_max
        # A first value with zero volt will be added
        v_max = int(np.round(csv_data[-1, 0]))
        v_interp = np.arange(v_max + 1)
        coss_interp = np.interp(v_interp, csv_data[:, 0], csv_data[:, 1])
        # Since we now have a evenly spaced vector where x corespond to the element-number of the vector
        # we dont have to store x (v_interp) with it.
        # To get Coss(V) just get the array element coss_interp[V]

        # np.savetxt('coss_' + name + '.csv', coss_interp, delimiter=';')

        # return coss_interp
        self['coss_' + name] = coss_interp
        self['qoss_' + name] = self._integrate_Coss(coss_interp)

    def _integrate_Coss(self, coss):
        """
        Integrate Coss for each voltage from 0 to V_max
        :param coss: MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
        :return: Qoss(Vds) as one row of data and index = Vds.
        """

        # Integrate from 0 to v
        def integrate(v):
            v_interp = np.arange(v + 1)
            coss_v = np.interp(v_interp, np.arange(coss.shape[0]), coss)
            return np.trapz(coss_v)

        coss_int = np.vectorize(integrate)
        # get an qoss vector that has the resolution 1V from 0 to V_max
        v_vec = np.arange(coss.shape[0])
        # get an qoss vector that fits the mesh_V scale
        # v_vec = np.linspace(V_min, V_max, int(V_step))
        qoss = coss_int(v_vec)
        # Scale from pC to nC
        # qoss = qoss / 1000

        # np.savetxt('qoss.csv', qoss, delimiter=';')
        return qoss


def save_to_file(dab: DAB_Data, directory=str(), name=str(), timestamp=True, comment=str()):
    """
    Save everything (except plots) in one file.
    WARNING: Existing files will be overwritten!

    File is ZIP compressed and contains several named np.ndarray objects:
        # String is constructed as follows:
        # used module (e.g. "mod_sps_") + value name (e.g. "phi")
        mod_sps_phi: mod_sps calculated values for phi
        mod_sps_tau1: mod_sps calculated values for tau1
        mod_sps_tau2: mod_sps calculated values for tau1
        sim_sps_iLs: simulation results with mod_sps for iLs
        sim_sps_S11_p_sw:

    :param comment:
    :param dab:
    :param directory: Folder where to save the files
    :param name: String added to the filename. Without file extension. Datetime may prepend the final name.
    :param timestamp: If the datetime should prepend the final name. default True
    """

    # Add some descriptive data to the file
    # Adding a timestamp, it may be useful
    dab['_timestamp'] = np.asarray(datetime.now().isoformat())
    # Adding a comment to the file, hopefully a descriptive one
    if comment:
        dab['_comment'] = np.asarray(comment)

    # Adding a timestamp to the filename if requested
    if timestamp:
        if name:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
        else:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        if name:
            filename = name
        else:
            # set some default non-empty filename
            filename = "dab_dataset"

    if directory:
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            file = os.path.join(directory, filename)
        else:
            warning("Directory does not exist!")
            file = os.path.join(filename)
    else:
        file = os.path.join(filename)

    # numpy saves everything for us in a handy zip file
    np.savez_compressed(file=file, **dab)


def load_from_file(file: str) -> DAB_Data:
    """
    Load everything from the given .npz file.
    :param file: a .nps filename or file-like object, string, or pathlib.Path
    :return: two objects with type DAB_Specification and DAB_Results
    """
    dab = DAB_Data()
    # Check for filename extension
    file_name, file_extension = os.path.splitext(file)
    if not file_extension:
        file += '.npz'
    file = os.path.expanduser(file)
    file = os.path.expandvars(file)
    file = os.path.abspath(file)
    # Open the file and parse the data
    with np.load(file) as data:
        for k, v in data.items():
            dab[k] = v
    return dab


def save_to_csv(dab: DAB_Data, key=str(), directory=str(), name=str(), timestamp=True):
    """
    Save one array with name 'key' out of dab_results to a csv file
    :param dab:
    :param key: name of the array in dab_results
    :param directory:
    :param name: filename without extension
    :param timestamp: if the filename should prepended with a timestamp
    """
    if timestamp:
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
    filename = key + '_' + name + '.csv'

    if directory:
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            file = os.path.join(directory, filename)
        else:
            warning("Directory does not exist!")
            file = os.path.join(filename)
    else:
        file = os.path.join(filename)

    comment = key + ' with P: {}, V1: {} and V2: {} steps.'.format(
        int(dab.P_step),
        int(dab.V1_step),
        int(dab.V2_step)
    )

    # Write the array to disk
    with open(file, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# ' + comment + '\n')
        outfile.write('# Array shape: {0}\n'.format(dab[key].shape))
        # x: P, y: V1, z(slices): V2
        outfile.write('# x: P ({}-{}), y: V1 ({}-{}), z(slices): V2 ({}-{})\n'.format(
            int(dab.P_min),
            int(dab.P_max),
            int(dab.V1_min),
            int(dab.V1_max),
            int(dab.V2_min),
            int(dab.V2_max)
        ))
        outfile.write('# z: V2 ' + np.array_str(dab.mesh_V2[:, 0, 0], max_line_width=10000) + '\n')
        outfile.write('# y: V1 ' + np.array_str(dab.mesh_V1[0, :, 0], max_line_width=10000) + '\n')
        outfile.write('# x: P ' + np.array_str(dab.mesh_P[0, 0, :], max_line_width=10000) + '\n')

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        i = 0
        for array_slice in dab[key]:
            # Writing out a break to indicate different slices...
            outfile.write('# V2 slice {}V\n'.format(
                (dab.V2_min + i * (dab.V2_max - dab.V2_min) / (dab.V2_step - 1))
            ))
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            # np.savetxt(outfile, array_slice, fmt='%-7.2f')
            np.savetxt(outfile, array_slice, delimiter=';')
            i += 1


def dab_ds_default() -> DAB_Data():
    ## Normal DAB
    # Set the basic DAB Specification
    dab = DAB_Data()
    dab.V1_nom = 700
    # dab.V1_min = 600
    # dab.V1_max = 800
    # dab.V1_step = 21
    # As requested only fixed V1
    dab.V1_min = 700
    dab.V1_max = 700
    dab.V1_step = 1
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = 25
    # dab.P_min = -2200
    # As requested only positive power transfer
    dab.P_min = 0
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 19
    dab.n = 2.99
    # ESB type not used in Gv5
    # dab.Ls = 83e-6
    # dab.Lm = 595e-6
    # Pi-ESB used in Gv5
    dab.Ls = 85e-6
    dab.Lc1 = 25.62e-3
    dab.Lc2 = 611e-6
    dab.fs = 200000
    # Switch dead time, must be >= t_sh_delay >= timestep_pre (equal is in both cases best)
    dab.t_dead1 = 50e-9
    dab.t_dead2 = 50e-9
    # Capacitance at Bridge Terminals assuming Side 1 and 2 are equal
    dab.C_PCB_leg = 5e-12
    # Junction Temperature of MOSFETs
    dab.temp = 150
    # Generate meshes
    dab.gen_meshes()

    # Import Coss curves
    # mosfet1 = 'C3M0120100J'
    mosfet1 = 'C3M0065100J'
    csv_file = '../Coss_files/Coss_' + mosfet1 + '.csv'
    dab.mosfet1 = mosfet1
    dab.import_Coss(csv_file, '1')
    # mosfet2 = 'C3M0120100J'
    mosfet2 = 'C3M0060065J'
    csv_file = '../Coss_files/Coss_' + mosfet2 + '.csv'
    dab.mosfet2 = mosfet2
    dab.import_Coss(csv_file, '2')

    return dab


def dab_ds_default_Gv6() -> DAB_Data():
    ## Normal DAB
    # Set the basic DAB Specification
    dab = DAB_Data()
    dab.V1_nom = 700
    # dab.V1_min = 600
    # dab.V1_max = 800
    # dab.V1_step = 21
    # As requested only fixed V1
    dab.V1_min = 700
    dab.V1_max = 700
    dab.V1_step = 1
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = 25
    # dab.P_min = -2200
    # As requested only positive power transfer
    dab.P_min = 0
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 19
    dab.n = 2.99
    # Pi-ESB used in Gv6
    dab.Ls = 110e-6
    dab.Lc1 = 25.62e-3
    dab.Lc2 = 611e-6 / (dab.n ** 2)
    dab.fs = 200000
    # Switch dead time, must be >= t_sh_delay >= timestep_pre (equal is in both cases best)
    dab.t_dead1 = 50e-9
    dab.t_dead2 = 50e-9
    # Capacitance parallel to MOSFETs
    dab.C_HB11 = 14.9e-12
    dab.C_HB12 = 22.9e-12
    dab.C_HB21 = 16.2e-12
    dab.C_HB22 = 13.7e-12
    # Junction Temperature of MOSFETs
    dab.temp = 150
    # Generate meshes
    dab.gen_meshes()

    # Import Coss curves
    # mosfet1 = 'C3M0120100J'
    mosfet1 = 'C3M0065100J'
    csv_file = '../Coss_files/Coss_' + mosfet1 + '.csv'
    dab.mosfet1 = mosfet1
    dab.import_Coss(csv_file, '1')
    # mosfet2 = 'C3M0120100J'
    mosfet2 = 'C3M0060065J'
    csv_file = '../Coss_files/Coss_' + mosfet2 + '.csv'
    dab.mosfet2 = mosfet2
    dab.import_Coss(csv_file, '2')

    return dab


def dab_ds_default_Gv7() -> DAB_Data():
    ## Normal DAB
    # Set the basic DAB Specification
    dab = DAB_Data()
    dab.V1_nom = 700
    # dab.V1_min = 600
    # dab.V1_max = 800
    # dab.V1_step = 21
    # As requested only fixed V1
    dab.V1_min = 700
    dab.V1_max = 700
    dab.V1_step = 1
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = 25
    # dab.P_min = -2200
    # As requested only positive power transfer
    dab.P_min = 0
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 19
    dab.n = 2.99
    # Pi-ESB used in Gv6
    # dab.Ls = 95e-6
    dab.Ls = 85e-6
    dab.Lc1 = 25.62e-3
    dab.Lc2 = 611e-6 / (dab.n ** 2)
    dab.fs = 200000
    # Switch dead time, must be >= t_sh_delay >= timestep_pre (equal is in both cases best)
    dab.t_dead1 = 50e-9
    dab.t_dead2 = 50e-9
    # Capacitance parallel to MOSFETs
    C_HB = 16e-12
    dab.C_HB11 = C_HB
    dab.C_HB12 = C_HB
    dab.C_HB21 = C_HB
    dab.C_HB22 = C_HB
    # Junction Temperature of MOSFETs
    dab.temp = 150
    # Generate meshes
    dab.gen_meshes()

    # Import Coss curves
    # mosfet1 = 'C3M0120100J'
    mosfet1 = 'C3M0065100J'
    csv_file = '../Coss_files/Coss_' + mosfet1 + '.csv'
    dab.mosfet1 = mosfet1
    dab.import_Coss(csv_file, '1')
    # mosfet2 = 'C3M0120100J'
    mosfet2 = 'C3M0060065J'
    csv_file = '../Coss_files/Coss_' + mosfet2 + '.csv'
    dab.mosfet2 = mosfet2
    dab.import_Coss(csv_file, '2')

    return dab


def dab_ds_default_Gv8() -> DAB_Data():
    ## Normal DAB
    # Set the basic DAB Specification
    dab = DAB_Data()
    dab.V1_nom = 700
    # dab.V1_min = 600
    # dab.V1_max = 800
    # dab.V1_step = 21
    # As requested only fixed V1
    dab.V1_min = 700
    dab.V1_max = 700
    dab.V1_step = 1
    dab.V2_nom = 235
    dab.V2_min = 175
    # dab.V2_min = 235
    dab.V2_max = 295
    dab.V2_step = 25
    # dab.V2_step = 7
    # dab.P_min = -2200
    # As requested only positive power transfer
    dab.P_min = 0
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 19
    # dab.P_step = 7
    dab.n = 2.99
    # Pi-ESB used in Gv6
    dab.Ls = 85e-6
    dab.Lc1 = 25.62e-3
    dab.Lc2 = 611e-6 / (dab.n ** 2)
    # dab.Lc1 = 500e-6
    # dab.Lc2 = 55e-6  # / (dab.n ** 2)
    dab.fs = 200000
    # Switch dead time, must be >= timestep_pre (equal is best)
    dab.t_dead1 = 100e-9
    dab.t_dead2 = 100e-9
    # Capacitance parallel to MOSFETs
    C_HB = 16e-12
    dab.C_HB11 = C_HB
    dab.C_HB12 = C_HB
    dab.C_HB21 = C_HB
    dab.C_HB22 = C_HB
    # Junction Temperature of MOSFETs
    dab.temp = 150
    # Generate meshes
    dab.gen_meshes()

    # Import Coss curves
    # mosfet1 = 'C3M0120100J'
    mosfet1 = 'C3M0065100J'
    csv_file = '../Coss_files/Coss_' + mosfet1 + '.csv'
    dab.mosfet1 = mosfet1
    dab.import_Coss(csv_file, '1')
    # mosfet2 = 'C3M0120100J'
    mosfet2 = 'C3M0060065J'
    csv_file = '../Coss_files/Coss_' + mosfet2 + '.csv'
    dab.mosfet2 = mosfet2
    dab.import_Coss(csv_file, '2')

    return dab


def dab_ds_default_Gv8_sim() -> DAB_Data():
    ## Normal DAB
    # Set the basic DAB Specification
    dab = DAB_Data()
    dab.V1_nom = 700
    # dab.V1_min = 600
    # dab.V1_max = 800
    # dab.V1_step = 21
    # As requested only fixed V1
    dab.V1_min = 700
    dab.V1_max = 700
    dab.V1_step = 1
    dab.V2_nom = 235
    # dab.V2_min = 175
    dab.V2_min = 235
    dab.V2_max = 295
    # dab.V2_step = 25
    dab.V2_step = 7
    # dab.P_min = -2200
    # As requested only positive power transfer
    dab.P_min = 400
    dab.P_max = 2200
    dab.P_nom = 2000
    # dab.P_step = 19
    dab.P_step = 7
    dab.n = 2.99
    # Pi-ESB used in Gv6
    dab.Ls = 85e-6
    dab.Lc1 = 25.62e-3
    dab.Lc2 = 611e-6 / (dab.n ** 2)
    # dab.Lc1 = 500e-6
    # dab.Lc2 = 55e-6  # / (dab.n ** 2)
    dab.fs = 200000
    # Switch dead time, must be >= timestep_pre (equal is best)
    dab.t_dead1 = 100e-9
    dab.t_dead2 = 100e-9
    # Capacitance parallel to MOSFETs
    C_HB = 16e-12
    dab.C_HB11 = C_HB
    dab.C_HB12 = C_HB
    dab.C_HB21 = C_HB
    dab.C_HB22 = C_HB
    # Junction Temperature of MOSFETs
    dab.temp = 150
    # Generate meshes
    dab.gen_meshes()

    # Import Coss curves
    # mosfet1 = 'C3M0120100J'
    mosfet1 = 'C3M0065100J'
    csv_file = '../Coss_files/Coss_' + mosfet1 + '.csv'
    dab.mosfet1 = mosfet1
    dab.import_Coss(csv_file, '1')
    # mosfet2 = 'C3M0120100J'
    mosfet2 = 'C3M0060065J'
    csv_file = '../Coss_files/Coss_' + mosfet2 + '.csv'
    dab.mosfet2 = mosfet2
    dab.import_Coss(csv_file, '2')

    return dab


def dab_ds_default_Gv8_sim_n4() -> DAB_Data():
    ## Normal DAB
    # Set the basic DAB Specification
    dab = DAB_Data()
    dab.V1_nom = 700
    # As requested only fixed V1
    dab.V1_min = 700
    dab.V1_max = 700
    dab.V1_step = 1
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    # dab.V2_step = 25
    dab.V2_step = 7
    # dab.P_min = -2200
    # As requested only positive power transfer
    dab.P_min = 400
    dab.P_max = 2200
    dab.P_nom = 2000
    # dab.P_step = 19
    dab.P_step = 7
    dab.n = 4
    # Pi-ESB used in Gv6
    dab.Ls = 85e-6
    dab.Lc1 = 25.62e-3
    dab.Lc2 = 611e-6 / (dab.n ** 2)
    # dab.Lc1 = 500e-6
    # dab.Lc2 = 55e-6  # / (dab.n ** 2)
    dab.fs = 200000
    # Switch dead time, must be >= timestep_pre (equal is best)
    dab.t_dead1 = 100e-9
    dab.t_dead2 = 100e-9
    # Capacitance parallel to MOSFETs
    C_HB = 16e-12
    dab.C_HB11 = C_HB
    dab.C_HB12 = C_HB
    dab.C_HB21 = C_HB
    dab.C_HB22 = C_HB
    # Junction Temperature of MOSFETs
    dab.temp = 150
    # Generate meshes
    dab.gen_meshes()

    # Import Coss curves
    # mosfet1 = 'C3M0120100J'
    mosfet1 = 'C3M0065100J'
    csv_file = '../Coss_files/Coss_' + mosfet1 + '.csv'
    dab.mosfet1 = mosfet1
    dab.import_Coss(csv_file, '1')
    # mosfet2 = 'C3M0120100J'
    mosfet2 = 'C3M0060065J'
    csv_file = '../Coss_files/Coss_' + mosfet2 + '.csv'
    dab.mosfet2 = mosfet2
    dab.import_Coss(csv_file, '2')

    return dab


def dab_ds_reversed() -> DAB_Data():
    ## Reversed DAB
    # Set the basic DAB Specification
    dab = DAB_Data()
    dab.V2_nom = 700
    dab.V2_min = 600
    dab.V2_max = 800
    dab.V2_step = 21
    dab.V1_nom = 235
    dab.V1_min = 175
    dab.V1_max = 295
    dab.V1_step = 25
    dab.P_min = -2200
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 19
    dab.n = 1 / 2.99
    dab.Ls = 83e-6 * dab.n ** 2
    dab.Lm = 595e-6 * dab.n ** 2
    # dab.Lc1 = 25.62e-3
    # dab.Lc1 = 800e-6
    # Assumption for tests
    dab.Lc1 = 611e-6 * dab.n ** 2
    dab.Lc2 = 611e-6 * dab.n ** 2
    # dab.Lc2 = 25e-3 * dab.n ** 2
    dab.fs = 200000
    dab.t_dead1 = 50e-9
    dab.t_dead2 = 50e-9
    # Generate meshes
    dab.gen_meshes()
    return dab


def dab_ds_everts() -> DAB_Data():
    ## DAB Everts
    # Set the basic DAB Specification
    dab = DAB_Data()
    dab.V1_nom = 250
    # dab.V1_min = 30
    dab.V1_min = 125
    dab.V1_max = 325
    # dab.V1_step = math.floor((dab.V1_max - dab.V1_min) / 10 + 1)  # 10V resolution gives 21 steps
    dab.V1_step = 21
    dab.V2_nom = 400
    dab.V2_min = 370
    dab.V2_max = 470
    # dab.V2_step = math.floor((dab.V2_max - dab.V2_min) / 10 + 1)  # 5V resolution gives 25 steps
    dab.V2_step = 21
    dab.P_min = -3700
    dab.P_max = 3700
    dab.P_nom = 2000
    # dab.P_step = math.floor((dab.P_max - dab.P_min) / 100 + 1)  # 100W resolution gives 19 steps
    dab.P_step = 38
    dab.n = 1
    dab.Ls = 13e-6
    dab.Lc1 = 62.1e-6
    dab.Lc2 = 62.1e-6
    dab.fs = 120e3
    dab.t_dead1 = 100e-9
    dab.t_dead2 = 100e-9
    # Generate meshes
    dab.gen_meshes()
    return dab


# FIXME These classes are deprecated and only for backward compability.

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


def old_save_to_file(dab_specs: DAB_Specification, dab_results: DAB_Results,
                     directory=str(), name=str(), timestamp=True, comment=str()):
    """
    Save everything (except plots) in one file.
    WARNING: Existing files will be overwritten!

    File is ZIP compressed and contains several named np.ndarray objects:
        # Starting with "dab_specs_" are for the DAB_Specification
        dab_specs_keys: containing the dict keys as strings
        dab_specs_values: containing the dict values as float
        # Starting with "dab_results_" are for the DAB_Results
        # TODO shoud I store generated meshes? Maybe not but regenerate them after loading a file.
        dab_results_mesh_V1: generated mesh
        dab_results_mesh_V2: generated mesh
        dab_results_mesh_P: generated mesh
        # String is constructed as follows:
        # "dab_results_" + used module (e.g. "mod_sps_") + value name (e.g. "phi")
        dab_results_mod_sps_phi: mod_sps calculated values for phi
        dab_results_mod_sps_tau1: mod_sps calculated values for tau1
        dab_results_mod_sps_tau2: mod_sps calculated values for tau1
        dab_results_sim_sps_iLs: simulation results with mod_sps for iLs
        dab_results_sim_sps_S11_p_sw:

    :param comment:
    :param dab_specs:
    :param dab_results:
    :param directory: Folder where to save the files
    :param name: String added to the filename. Without file extension. Datetime may prepend the final name.
    :param timestamp: If the datetime should prepend the final name. default True
    """
    # Temporary Dict to hold the name/array (key/value) pairs to be stored by np.savez
    # Arrays to save to the file. Each array will be saved to the output file with its corresponding keyword name.
    kwds = dict()
    kwds['dab_specs_keys'], kwds['dab_specs_values'] = dab_specs.export_to_array()

    for k, v in dab_results.items():
        # TODO filter for mesh here
        kwds['dab_results_' + k] = v

    # Add some descriptive data to the file
    # Adding a timestamp, it may be useful
    kwds['_timestamp'] = np.asarray(datetime.now().isoformat())
    # Adding a comment to the file, hopefully a descriptive one
    if comment:
        kwds['_comment'] = np.asarray(comment)

    # Adding a timestamp to the filename if requested
    if timestamp:
        if name:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
        else:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        if name:
            filename = name
        else:
            # set some default non-empty filename
            filename = "dab_opt_dataset"

    if directory:
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            file = os.path.join(directory, filename)
        else:
            warning("Directory does not exist!")
            file = os.path.join(filename)
    else:
        file = os.path.join(filename)

    # numpy saves everything for us in a handy zip file
    # np.savez(file=file, **kwds)
    np.savez_compressed(file=file, **kwds)


def old_load_from_file(file: str) -> tuple[DAB_Specification, DAB_Results]:
    """
    Load everything from the given .npz file.
    :param file: a .nps filename or file-like object, string, or pathlib.Path
    :return: two objects with type DAB_Specification and DAB_Results
    """
    dab_specs = DAB_Specification()
    dab_results = DAB_Results()
    # Open the file and parse the data
    with np.load(file) as data:
        spec_keys = None
        spec_values = None
        for k, v in data.items():
            if k.startswith('dab_results_'):
                dab_results[k.removeprefix('dab_results_')] = v
            if str(k) == 'dab_specs_keys':
                spec_keys = v
            if str(k) == 'dab_specs_values':
                spec_values = v
            if str(k) == '_timestamp':
                dab_results[k] = v
            if str(k) == '_comment':
                dab_results[k] = v
        # We can not be sure if specs where in the file but hopefully there was
        if not (spec_keys is None and spec_values is None):
            dab_specs.import_from_array(spec_keys, spec_values)
        # TODO regenerate meshes here
    return dab_specs, dab_results


def old_save_to_csv(dab_specs: DAB_Specification, dab_results: DAB_Results, key=str(), directory=str(), name=str(),
                    timestamp=True):
    """
    Save one array with name 'key' out of dab_results to a csv file
    :param dab_specs:
    :param dab_results:
    :param key: name of the array in dab_results
    :param directory:
    :param name: filename without extension
    :param timestamp: if the filename should prepended with a timestamp
    """
    if timestamp:
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
    filename = os.path.join(directory + '/' + name + '_' + key + '.csv')

    if directory:
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if os.path.isdir(directory):
            file = os.path.join(directory, filename)
        else:
            warning("Directory does not exist!")
            file = os.path.join(filename)
    else:
        file = os.path.join(filename)

    comment = key + ' with P: {}, V1: {} and V2: {} steps.'.format(
        int(dab_specs.P_step),
        int(dab_specs.V1_step),
        int(dab_specs.V2_step)
    )

    # Write the array to disk
    with open(file, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# ' + comment + '\n')
        outfile.write('# Array shape: {0}\n'.format(dab_results[key].shape))
        # x: P, y: V1, z(slices): V2
        outfile.write('# x: P ({}-{}), y: V1 ({}-{}), z(slices): V2 ({}-{})\n'.format(
            int(dab_specs.P_min),
            int(dab_specs.P_max),
            int(dab_specs.V1_min),
            int(dab_specs.V1_max),
            int(dab_specs.V2_min),
            int(dab_specs.V2_max)
        ))
        outfile.write('# z: V2 ' + np.array_str(dab_results.mesh_V2[:, 0, 0], max_line_width=10000) + '\n')
        outfile.write('# y: V1 ' + np.array_str(dab_results.mesh_V1[0, :, 0], max_line_width=10000) + '\n')
        outfile.write('# x: P ' + np.array_str(dab_results.mesh_P[0, 0, :], max_line_width=10000) + '\n')

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        i = 0
        for array_slice in dab_results[key]:
            # Writing out a break to indicate different slices...
            outfile.write('# V2 slice {}V\n'.format(
                (dab_specs.V2_min + i * (dab_specs.V2_max - dab_specs.V2_min) / (dab_specs.V2_step - 1))
            ))
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            # np.savetxt(outfile, array_slice, fmt='%-7.2f')
            np.savetxt(outfile, array_slice, delimiter=';')
            i += 1


def old_to_new(dab_specs: DAB_Specification, dab_results: DAB_Results) -> DAB_Data:
    """
    Converts the two old datasets to the new combined dataset
    :param dab_specs:
    :param dab_results:
    :return:
    """
    dab = DAB_Data()
    for k, v in dab_specs.items():
        dab[k] = v
    for k, v in dab_results.items():
        dab[k] = v
    return dab


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module Datasets ...")

    # Instantiate the DAB_Data class that holds all our DAB related data
    dab = DAB_Data()

    # Set the basic DAB Specification
    dab.V1_nom = 700
    dab.V1_min = 600
    dab.V1_max = 800
    dab.V1_step = 3
    dab.V2_nom = 235
    dab.V2_min = 175
    dab.V2_max = 295
    dab.V2_step = 4
    dab.P_min = 400
    dab.P_max = 2200
    dab.P_nom = 2000
    dab.P_step = 5
    dab.n = 2.99
    dab.L_s = 84e-6
    dab.L_m = 599e-6
    dab.fs_nom = 200000
    debug(dab)

    # Test mesh generation
    dab.gen_meshes()

    # Test if single value entries work as expected even so they are ndarrays now
    step = int((dab.V1_max - dab.V1_min) / 10 + 1)
    debug(step)

    # Test save dab
    debug(dab)
    save_to_file(dab, name='test_dab_data', timestamp=False)

    # Test load dab
    dab2 = load_from_file('test_dab_data')
    debug(dab2)

    # Test csv export
    save_to_csv(dab, 'mesh_V1', name='test_dab_csv', timestamp=False)

    # OLD Deprecated classes:

    # # Set the basic DAB Specification
    # # Setting it this way disables tab completion!
    # # Don't use this!
    # dab_test_dict = {'V1_nom':  700,
    #                  'V1_min':  600,
    #                  'V1_max':  800,
    #                  'V1_step': 3,
    #                  'V2_nom':  235,
    #                  'V2_min':  175,
    #                  'V2_max':  295,
    #                  'V2_step': 3,
    #                  'P_min':   400,
    #                  'P_max':   2200,
    #                  'P_nom':   2000,
    #                  'P_step':  3,
    #                  'n':       2.99,
    #                  'L_s':     84e-6,
    #                  'L_m':     599e-6,
    #                  'fs_nom':  200000
    #                  }
    # dab_test_dm_no_completion = DAB_Specification(dab_test_dict)
    # # Check Value types
    # for value in dab_test_dm_no_completion.values():
    #     print(type(value))
    #
    # # Set the basic DAB Specification
    # # Setting it this way enables tab completion!
    # dab_test_dm = DAB_Specification()
    # dab_test_dm.V1_nom = 700
    # dab_test_dm.V1_min = 600
    # dab_test_dm.V1_max = 800
    # dab_test_dm.V1_step = 3
    # dab_test_dm.V2_nom = 235
    # dab_test_dm.V2_min = 175
    # dab_test_dm.V2_max = 295
    # dab_test_dm.V2_step = 3
    # dab_test_dm.P_min = 400
    # dab_test_dm.P_max = 2200
    # dab_test_dm.P_nom = 2000
    # dab_test_dm.P_step = 3
    # dab_test_dm.n = 2.99
    # dab_test_dm.L_s = 84e-6
    # dab_test_dm.L_m = 599e-6
    # dab_test_dm.fs_nom = 200000
    #
    # # Some DotMap access examples
    # print(dab_test_dm.V1_nom)
    # print(dab_test_dm['V2_nom'])
    # print(dab_test_dm)
    # print(dab_test_dm.toDict())
    # dab_test_dm.pprint()
    # dab_test_dm.pprint(pformat='json')
    # # Check Value types
    # for value in dab_test_dm.values():
    #     print(type(value))
    #
    # # export
    # print("export")
    # spec_keys, spec_values = dab_test_dm.export_to_array()
    # print(spec_keys, spec_values)
    # # import
    # print("import")
    # dab_loaded = DAB_Specification()
    # dab_loaded.import_from_array(spec_keys, spec_values)
    # print(dab_loaded)
