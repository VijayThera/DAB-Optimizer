#!/usr/bin/python3
# coding: utf-8
# python >= 3.10

import os
import numpy as np
from datetime import datetime, timezone

import classes_datasets as ds
import debug_tools as db
import mod_cpm
import sim_gecko
import plot_dab

from plotWindow import plotWindow


def save_to_file(dab_specs: ds.DAB_Specification, dab_results: ds.DAB_Results,
                 directory = str(), name = str(), timestamp = True, comment = str()):
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
        # "dab_results_" + used module (e.g. "mod_cpm_") + value name (e.g. "phi")
        dab_results_mod_cpm_phi: mod_cpm calculated values for phi
        dab_results_mod_cpm_tau1: mod_cpm calculated values for tau1
        dab_results_mod_cpm_tau2: mod_cpm calculated values for tau1
        dab_results_sim_cpm_iLs: simulation results with mod_cpm for iLs
        dab_results_sim_cpm_S11_p_sw:

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

    if (not directory) or os.path.isdir(directory):
        file = os.path.join(directory, filename)
    else:
        print("Directory does not exist!")
        return

    # numpy saves everything for us in a handy zip file
    #np.savez(file=file, **kwds)
    np.savez_compressed(file=file, **kwds)

def load_from_file(file: str) -> tuple[ds.DAB_Specification, ds.DAB_Results]:
    """
    Load everything from the given .npz file.
    :param file: a .nps filename or file-like object, string, or pathlib.Path
    :return: two objects with type DAB_Specification and DAB_Results
    """
    dab_specs = ds.DAB_Specification()
    dab_results = ds.DAB_Results()
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


@db.timeit
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
    # TODO provide a generator function for this in the DAB_Results class
    dab_results.mesh_V1, dab_results.mesh_V2, dab_results.mesh_P = np.meshgrid(
        np.linspace(dab_specs.V1_min, dab_specs.V1_max, int(dab_specs.V1_step)),
        np.linspace(dab_specs.V2_min, dab_specs.V2_max, int(dab_specs.V2_step)),
        np.linspace(dab_specs.P_min, dab_specs.P_max, int(dab_specs.P_step)), sparse=False)

    # Modulation Calculation
    # TODO how to name the arrays according to some kind of given pattern?
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
    #pw.show()

    # Saving
    #save_to_file(dab_specs, dab_results, name='test-save', comment='This is a saving test with random data!')
    save_to_file(dab_specs, dab_results, name='test-save', timestamp=False, comment='This is a saving test with random data!')

    # Loading
    dab_specs_loaded, dab_results_loaded = load_from_file('test-save.npz')
    dab_specs_loaded.pprint()
    dab_results_loaded.pprint()


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of DAB Optimizer ...")

    main()

# sys.exit(0)
