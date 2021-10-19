#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import yaml


def generate_build_config():
    targets = {
        f"{p}_{avx}_{math}_build": {
            "extends": [f".{math}_image", f".{p}_{avx}", ".build"]
        }
        for p in ["cpu", "gpu"]
        for avx in ["none", "avx"]
        for math in ["mkl", "oss"]
    }
    print(yaml.dump(targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["build"])
    args = parser.parse_args()

    if args.target == "build":
        generate_build_config()
