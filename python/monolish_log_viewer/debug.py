"""log debug."""

import datetime

def add_print(pattern):
    """log Decorator"""
    def trace(func):
        # pylint: disable=unused-argument
        def wrapper(*args, **kwargs):
            dt_now = datetime.datetime.now()
            function_name = ','.join(args)
            list_text = ','.join(kwargs)
            print(f"[{dt_now}] {pattern} {function_name} {list_text}")
        return wrapper
    return trace

@add_print("")
def log_general(message: str) -> str:
    """usually log"""
    return message

@add_print("success")
def log_success(message: str) -> str:
    """success log"""
    return message

@add_print("error")
def log_error(message: str) -> str:
    """error log"""
    return message
