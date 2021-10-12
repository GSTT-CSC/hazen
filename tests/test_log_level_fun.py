import logging
from log_levels_fun import log

def multiply(x, y):
    log(f"Params passed to `multiply`: {x} and {y}")
    return x * y

ret = multiply(10, 2)