import pytest
import os
import sys
import pandas
import argparse

from monolish_log_viewer import __main__

def test_execute_create_log():
    # pass command line arguments
    sys.argv.append("./tests/test_data/cg_iter.yml")
    sys.argv.append("./tests/test_data/cg_iter.html")
    log_path = sys.argv[3]
    out_path = sys.argv[4]

    __main__.execute_create_log(log_path, out_path)