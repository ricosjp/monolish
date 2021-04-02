# Installation guide {#install_guide}
use apt command ....

## Intel CPU
coming soon...

## AMD CPU
coming soon...

## Intel CPU + NVIDIA GPU
coming soon...

## AMD CPU + NVIDIA GPU
coming soon...

## ARM
coming soon...

## Intel CPU + NVIDIA GPU
coming soon...

## Other architectures
### SX-Aurora TSUBASA 
We have confirmed that it works with past versions, but not with the latest version.

The Makefile that worked in the past version is [here](https://github.com/ricosjp/monolish/blob/master/Makefile.sxat).

### A64fx
We have confirmed that it works with past versions, but not with the latest version.

The Makefile that worked in the past version is [here](https://github.com/ricosjp/monolish/blob/master/Makefile.a64fx).

### IBM Power
Operation has not been confirmed due to lack of operating environment.
Our expectation is that it should work.

If anyone can confirm that it works, please report it to Issue.

### AMD Radeon GPU and Intel Xe GPU
Currently, it does not work because the code depends on NVIDIA CUDA libraries.

OpenMP Offloading should work on these architecture, so it should be possible to make it work based on the design concept of monolish.

These architectures will be supported in the future.
