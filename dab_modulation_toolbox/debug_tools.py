#!/usr/bin/python3
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
from datetime import datetime
# Decorator to print function args
import inspect
# Decorator to measure execution time of a function
from functools import wraps
import time

# Set a global DEBUG variable to switch some debugging code.
# This is evaluated a runtime, not like the Python __debug__ that is evaluated in preprocess.
DEBUG = False


class log:
    """
    Class to print logging text to stdout and to a log file.
    """
    logfile = None

    def __init__(self, filename=str()):
        if filename:
            filename = os.path.expanduser(filename)
            filename = os.path.expandvars(filename)
            filename = os.path.abspath(filename)
            self.logfile = open(filename, 'a', buffering=1)

    def __del__(self):
        self.close()

    def close(self):
        if self.logfile:
            self.logfile.close()

    def error(self, *args, sep='\n', **kwargs):
        """
        Log error output like print does.
        """
        # print(*args, **kwargs)
        print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(
            inspect.stack()[1][0]).__name__ + ' ' + sep.join(map(str, args)), **kwargs)
        if self.logfile:
            print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(
                inspect.stack()[1][0]).__name__ + ' ' + sep.join(map(str, args)), **kwargs, file=self.logfile)

    def warning(self, *args, sep='\n', **kwargs):
        """
        Log warning output like print does.
        """
        # print(*args, **kwargs)
        print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(
            inspect.stack()[1][0]).__name__ + ' ' + sep.join(map(str, args)), **kwargs)
        if self.logfile:
            print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(
                inspect.stack()[1][0]).__name__ + ' ' + sep.join(map(str, args)), **kwargs, file=self.logfile)

    def info(self, *args, sep='\n', **kwargs):
        """
        Log normal info output like print does.
        """
        print(*args, **kwargs, sep=sep)
        if self.logfile:
            print(*args, **kwargs, sep=sep, file=self.logfile)

    def debug(self, *args, sep='\n', **kwargs):
        """
        Log debug output like print does.
        """
        if DEBUG or __debug__:
            # highly detailed output
            print(datetime.now().isoformat(timespec='milliseconds') + ' '
                  + inspect.getmodule(inspect.stack()[1][0]).__name__ + ' '
                  + inspect.currentframe().f_back.f_code.co_name + '\n'
                  + sep.join(map(str, args)), **kwargs)
            if self.logfile:
                print(datetime.now().isoformat(timespec='milliseconds') + ' '
                      + inspect.getmodule(inspect.stack()[1][0]).__name__ + ' '
                      + inspect.currentframe().f_back.f_code.co_name + '\n'
                      + sep.join(map(str, args)), **kwargs, file=self.logfile)


def dump_args(func):
    """
    Decorator to print function call details.
    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"{func.__module__}.{func.__qualname__} ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper


# Use this in front of a function to print args
# @dump_args


def timeit(func):
    """
    Decorator to measure execution time of a function
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        # print(f'{total_time:.4f} seconds for Function {func.__name__}{args} {kwargs}')
        print(f'{total_time:.4f} seconds for Function {func.__name__}')
        return result

    return timeit_wrapper


# Use this in front of a function to measure execution time
# @timeit


def error(*args, sep='\n', **kwargs):
    """
    Log error output like print does.
    :param args:
    :param sep:
    :param kwargs:
    """
    # print(*args, **kwargs)
    print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ +
          ' ' + sep.join(map(str, args)), **kwargs)


def warning(*args, sep='\n', **kwargs):
    """
    Log warning output like print does.
    :param args:
    :param sep:
    :param kwargs:
    """
    # print(*args, **kwargs)
    print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ +
          ' ' + sep.join(map(str, args)), **kwargs)


def info(*args, sep='\n', **kwargs):
    """
    Log normal info output like print does.
    :param args:
    :param sep:
    :param kwargs:
    """
    print(*args, **kwargs)
    # print(datetime.now().isoformat(timespec='milliseconds') + ' ' + sep.join(map(str, args)), **kwargs)
    # print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ +
    #       ' ' + sep.join(map(str, args)), **kwargs)


def debug(*args, sep='\n', **kwargs):
    """
    Log debug output like print does.
    :param args:
    :param sep:
    :param kwargs:
    """
    if DEBUG or __debug__:
        # print(*args, **kwargs)
        # print(datetime.now().isoformat(timespec='milliseconds') + ' ' + sep.join(map(str, args)), **kwargs)
        # detailed output
        # print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ +
        #       ' ' + sep.join(map(str, args)), **kwargs)
        # highly detailed output
        print(datetime.now().isoformat(timespec='milliseconds') + ' ' +
              inspect.getmodule(inspect.stack()[1][0]).__name__ + ' ' +
              inspect.currentframe().f_back.f_code.co_name + '\n' +
              sep.join(map(str, args)), **kwargs)


# ---------- MAIN ----------
if __name__ == '__main__':
    print("Start of Module Debug ...")
