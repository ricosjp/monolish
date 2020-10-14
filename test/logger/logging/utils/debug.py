#!/usr/bin/env python3
# coding: utf-8

import datetime

"""log Decorator"""
def add_print(pattern):
    def trace(func):
        def wrapper(*args):
            dt_now = datetime.datetime.now()
            function_name = ','.join(args)
            print(f"[{dt_now}] {pattern} {function_name}")
        return wrapper
    return trace

@add_print("")
def log_general(message) -> str:
    return(message)

@add_print("success")
def log_success(message) -> str:
    return(message)

@add_print("error")
def log_error(message) -> str:
    return(message)

def dump(dict_list) -> str:
    dt_now = datetime.datetime.now()
    print(f"[{dt_now}] dump dict")
    for one_dict in dict_list:
        print(f"{one_dict}")