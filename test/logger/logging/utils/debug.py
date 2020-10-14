#!/usr/bin/env python3
# coding: utf-8

import datetime

"""log Decorator"""
def add_print(pattern):
    def trace(func):
        def wrapper(*args, **kwargs):
            dt_now = datetime.datetime.now()
            function_name = ','.join(args)
            list_text = ','.join(kwargs)
            print(f"[{dt_now}] {pattern} {function_name} {list_text}")
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