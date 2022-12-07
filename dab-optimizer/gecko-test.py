#!/usr/bin/python3
# -*- coding: utf-8 -*-
### python >= 3.10 ###


import os
#import sys
import argparse
import pathlib
import sys

import leapythontoolbox as lpt


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

    simfilepath = 'DAB_MOSFET_Modulation_Lm_nlC.ipes'

    if isinstance(simfilepath, str) and simfilepath.endswith('.ipes') and pathlib.Path(simfilepath).exists():
        print("geht")
        print(os.path.abspath(simfilepath))
    else:
        print("error")

    sys.exit(0)



    dab_converter = lpt.GeckoSimulation('DAB_MOSFET_Modulation_Lm_nlC.ipes', timestep=50e-9, simtime=15e-6)

    params = dab_converter.get_global_parameters(['phi', 'tau1_inv', 'tau2_inv'])
    print(params)
    params = {'phi': 80.0, 'tau1_inv': 40.0, 'tau2_inv': 66.0}
    dab_converter.set_global_parameters(params)

    dab_converter.run_simulation()

    dab_converter.get_scope_data(node_names=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], file_name='test')

    values = dab_converter.get_values(nodes=['v1', 'v2_1', 'i_HF1', 'S11_p_sw'], operations=['mean', 'rms'], range_start_stop=[10e-6, 15e-6])
    print(values)
