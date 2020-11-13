import pytest
import os
import sys
import pandas
import argparse

from monolish_log_viewer import __main__

def test_execute_create_log_normaldata():
    for any_file in ["only_solver", "normal_data", "only_other", "cg_iter"]:
        # pass command line arguments
        sys.argv.append(f"./tests/test_data/{any_file}.yml")
        sys.argv.append(f"./tests/test_data/{any_file}.html")
        log_path = sys.argv[-2]
        out_path = sys.argv[-1]

        write_number_of_character = __main__.execute_create_log(log_path, out_path)
        print("=================")

        assert isinstance(write_number_of_character, int) == True