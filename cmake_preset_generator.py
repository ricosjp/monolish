#!/usr/bin/python
# -*- coding: utf-8 -*-

import json


def display_name(p, avx, mpi, package):
    s = "monolish with " + p.upper()
    if avx == "avx":
        s += ", AVX"
    if mpi == "mpi":
        s += ", MPI"
    if package:
        s += " for packaging"
    return s


def cache_variables(p, avx, mpi, package):
    return {
        "MONOLISH_USE_AVX": "ON" if avx == "avx" else "OFF",
        "MONOLISH_USE_MPI": "ON" if mpi == "mpi" else "OFF",
        "MONOLISH_USE_NVIDIA_GPU": "ON" if p == "gpu" else "OFF",
        "MONOLISH_NVIDIA_GPU_ARCH_ALL": "ON" if package else "OFF",
    }


def base_dir(package):
    if package:
        return "${sourceDir}/package/"
    else:
        return "${sourceDir}/build/"


def main():
    targets = [
        {
            "name": f"{p}-{avx}-{mpi}{package}",
            "generator": "Ninja",
            "displayName": display_name(p, avx, mpi, package),
            "binaryDir": base_dir(package) + f"{p}-{avx}-{mpi}",
            "cacheVariables": cache_variables(p, avx, mpi, package),
        }
        for p in ["cpu", "gpu"]
        for avx in ["none", "avx"]
        for mpi in ["none", "mpi"]
        for package in ["", "-package"]
    ]
    preset = {
        "version": 2,
        "cmakeMinimumRequired": {"major": 3, "minor": 20, "patch": 0},
        "configurePresets": targets,
    }
    print(json.dumps(preset, indent=2))


if __name__ == "__main__":
    main()
