import pytest
import os
import sys
import pandas as pd
import argparse

from monolish_log_viewer import __main__

def test_main():
    # call with arg1 = "./tests/test_data/base_data3.yml", arg2 = "./tests/test_data/base_data3.html"
    res = __main__.main()
    poetry run monolish_log_viewer ./tests/test_data/base_data3.yml ./tests/test_data/base_data3.html