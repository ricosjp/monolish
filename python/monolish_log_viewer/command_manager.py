""" command_manager """
import argparse

def controll_argument():
    """ command line arguments """
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("out_path")

    args = parser.parse_args()
    log_path = args.log_path
    out_path = args.out_path

    return log_path, out_path
