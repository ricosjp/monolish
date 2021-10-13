#!/usr/bin/python
# -*- coding: utf-8 -*-

import json


def display_name(p, avx, mpi, arch):
    s = "monolish with " + p.upper()
    if arch:
        s += " for all CUDA arch"
    if avx == "avx":
        s += ", AVX"
    if mpi == "mpi":
        s += ", MPI"
    return s


def cache_variables(p, avx, mpi, arch):
    return {
        "MONOLISH_USE_AVX": "ON" if avx == "avx" else "OFF",
        "MONOLISH_USE_MPI": "ON" if mpi == "mpi" else "OFF",
        "MONOLISH_USE_NVIDIA_GPU": "ON" if p == "gpu" else "OFF",
        "MONOLISH_NVIDIA_GPU_ARCH_ALL": "ON" if arch else "OFF",
    }


def main():
    targets = [
        {
            "name": f"{p}-{avx}-{mpi}{arch}",
            "generator": "Ninja",
            "displayName": display_name(p, avx, mpi, arch),
            "binaryDir": "${sourceDir}/build/" + f"{p}-{avx}-{mpi}{arch}",
            "cacheVariables": cache_variables(p, avx, mpi, arch),
        }
        for p in ["cpu", "gpu"]
        for avx in ["none", "avx"]
        for mpi in ["none", "mpi"]
        for arch in ["", "-all"]
        if not (p == "cpu" and arch == "-all")
    ]
    preset = {
        "version": 2,
        "cmakeMinimumRequired": {"major": 3, "minor": 20, "patch": 0},
        "configurePresets": targets,
    }
    print(json.dumps(preset, indent=2))


if __name__ == "__main__":
    main()
