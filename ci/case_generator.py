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


def generate_package_config():
    targets = {
        f"gpu_sm_{sm}_{math}_build": {
            "extends": [f".{math}_image", ".package"],
            "variables": {
                "PRESET": "gpu-avx-none",
                "MONOLISH_NVIDIA_GPU_ARCH": f"sm_{sm}",
            },
        }
        for math in ["mkl", "oss"]
        for sm in ["52", "60", "61", "70", "75", "80"]
    }
    print(yaml.dump(targets))


def generate_docker_config():
    targets = {
        f"{math}_nvidia_sm_{sm}_docker": {
            "extends": [".docker"],
            "needs": [f"gpu_sm_{sm}_{math}_build"],
        }
        for math in ["mkl", "oss"]
        for sm in ["52", "60", "61", "70", "75", "80"]
    }
    print(yaml.dump(targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["build", "package", "docker"])
    args = parser.parse_args()
    {
        "build": generate_build_config,
        "package": generate_package_config,
        "docker": generate_docker_config,
    }[args.target]()
