#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import yaml


def generate_build_config():
    targets = {
        f"{p}_{avx}{compiler}_{math}_build": {
            "extends": [f".{math}_image", f".{p}_{avx}{compiler}", ".build"]
        }
        for p in ["cpu", "gpu"]
        for avx in ["none", "avx"]
        for math in ["mkl", "oss"]
        for compiler in ["", "_gcc"]
        if not (p == "gpu" and compiler == "_gcc")
    }
    print(yaml.dump(targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["build"])
    args = parser.parse_args()

    if args.target == "build":
        generate_build_config()
