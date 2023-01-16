#!/usr/bin/python3
# coding: utf-8
# python >= 3.10


from datetime import datetime
# Decorator to print function args
import inspect
# Decorator to measure execution time of a function
from functools import wraps
import time

# Set a global DEBUG variable to switch some debugging code.
# This is evaluated a runtime, not like the Python __debug__ that is evaluated in preprocess.
DEBUG = True


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


def error(*args, sep=' ', **kwargs):
    """
    Log error output like print does.
    :param args:
    :param sep:
    :param kwargs:
    """
    # print(*args, **kwargs)
    print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ +
          ' ' + sep.join(map(str, args)), **kwargs)


def warning(*args, sep=' ', **kwargs):
    """
    Log warning output like print does.
    :param args:
    :param sep:
    :param kwargs:
    """
    # print(*args, **kwargs)
    print(datetime.now().isoformat(timespec='milliseconds') + ' ' + inspect.getmodule(inspect.stack()[1][0]).__name__ +
          ' ' + sep.join(map(str, args)), **kwargs)


def info(*args, sep=' ', **kwargs):
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


def debug(*args, sep=' ', **kwargs):
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
