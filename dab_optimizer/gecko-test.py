#!/usr/bin/python3
# coding: utf-8
# python >= 3.10


import os
import pathlib
# import sys
# import argparse
# import sys

import pygeckocircuits2 as pgc

# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Gecko Simulation Test...")
    # Define arguments passed to the script
    # parser = argparse.ArgumentParser()
    # parser.add_argument("configfile", help="config file")
    # parser.add_argument("-l", help="Log to file: <datetime>_<name>.log")
    # parser.add_argument("-d", help="Set log output to debug level", action="store_true")
    # args = parser.parse_args()
    #
    # if os.path.isfile(args.configfile):
    #     config = yaml.load(open(args.configfile))
    # else:
    #     print("[ERROR] configfile '{}' does not exist!".format(args.configfile), file=sys.stderr)
    #     sys.exit(1)
    #
    # if args.d:
    #     loglevel = logging.DEBUG
    # else:
    #     loglevel = logging.INFO

    simfilepath = '../circuits/DAB_MOSFET_Modulation_Lm_nlC.ipes'

    if isinstance(simfilepath, str) and simfilepath.endswith('.ipes') and pathlib.Path(simfilepath).exists():
        print("geht")
        print(os.path.abspath(simfilepath))
    else:
        print("error")

    # sys.exit(0)

    print("pgc.GeckoSimulation(simfilepath)")
    # this opens the GUI
    dab_converter = pgc.GeckoSimulation(simfilepath)

    params = dab_converter.get_global_parameters(['phi', 'tau1_inv', 'tau2_inv', 'v_dc1', 'v_dc2', 'f_s'])
    print(params)
    params = {'phi': 80.0, 'tau1_inv': 40.0, 'tau2_inv': 66.0}
    dab_converter.set_global_parameters(params)

    print("dab_converter.run_simulation")
    dab_converter.run_simulation(timestep=50e-9, simtime=15e-6)

    dab_converter.get_scope_data(node_names=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], file_name='test')

    values = dab_converter.get_values(nodes=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], operations=['mean', 'rms'],
                                      range_start_stop=[10e-6, 15e-6])
    print(values)
    print(type(values))
    print(type(values['mean']['i_HF1']))
