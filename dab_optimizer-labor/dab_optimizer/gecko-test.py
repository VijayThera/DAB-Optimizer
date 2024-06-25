#!/usr/bin/python3
# coding: utf-8
# python >= 3.10


import os
import pathlib
# import sys
# import argparse
# import sys

import pygeckocircuits2 as lpt

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

    print("lpt.GeckoSimulation(simfilepath)")
    # this opens the GUI
    dab_converter = lpt.GeckoSimulation(simfilepath, simtime=0.05, timestep=50e-9, simtime_pre=100e-3, timestep_pre=20e-9)

    dab_converter.get_global_parameters(['phi', 'tau1_inv', 'tau2_inv', 'v_dc1', 'v_dc2', 'f_s'])

    params = {'n': 4, 'v_dc1': 700, 'v_dc2': 175, 'f_s': 200000, 'phi': 90, 'tau1_inv': 45, 'tau2_inv': 45}

    print(params)

    dab_converter.set_global_parameters(params)

    print("dab_converter.run_simulation")
    dab_converter.run_simulation(save_file=True)

    dab_converter.get_scope_data(node_names=['i_HF1'], file_name='scope_data')

    #values = dab_converter.get_values(nodes=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], operations=['mean', 'rms'],
    #                                  range_start_stop=[100e-6, 500e-6])
    values = dab_converter.get_values(nodes=['i_HF1'],
                                      operations=['rms'],
                                      range_start_stop=[110e-3, 120e-3])

    print('---------------------------------------------------')
    # print(values['mean'])
    # print(values)
    # print(type(values['i_HF1']))
