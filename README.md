# monolish: MONOlithic Liner equation Solvers for Highly-parallel architecture
monolish is a linear solver library that monolithically fuses variable data type, matrix structures, matrix data format, vender specific data transfer APIs, and vender specific numerical algebra libraries.

# Feature
monolish let developper be oblivious about:
- [Performance tuning][tuning_rule]
- [Various processor which execute library  (Intel CPU / NVIDIA GPU / AMD CPU / ARM CPU / NEC SX-Aurora TSUBASA etc.) ][oplist]
- [Vender specific data transfer API (host RAM to Device RAM)][gpu]
- [Find bottolenecks][logger] and [Perormance benchmark][perfviewer]
- Argument data type of matrix / vector operations ([doxygen function list][doxy_func])
- [Matrix structure / storage format][matrix_storage]
- [Cumbersome package dependency][build]

[oplist]: doc/operation_list.md
[gpu]: doc/gpu.md
[doxy_func]: XXXXXXXXXXXXXXX
[build]: doc/build.md
[logger]: XXXXXXXXXXXXXXXXXXXXXXXXX
[perfviewer]: XXXXXXXXXXXXXXXXXX
[matrix_storage]: XXXXXXXXXXXXXXXX
[tunenig_rule]: XXXXXXXXXXXXXXX 

# What monolish solves
monolish is a vendor-independent open source library written in C++ that aims to be grand unified linear algebra library on any hardware.

BLAS has 150+ functions and lapack has 1000+ functions. These are complete software for dense matrix operations. 
However, these are not enough due to the growing needs of users. 
In particular, sparse matrix operations and machine learning kernels are not implemented. 

To solve this, the libraries (MKL, cusolver, etc.) by each hardware vendor and numpy / scipy, julia, matlab, etc. implement extended functions.
On the other hand, these software APIs are not unified. These are language or vecder specific.

monolish provides an API that integrates the numerical algebra libraries of each vendor.
monolish calls the vendor-developed numerical algebra libraries whenever possible.
monolish implements and provides functions that are not implemented in these libraries.

monolish solves cumbersome package management by Docker.

# Build and Install
## Download binary
XXXXXXX

## Build (for monolish Developpers)
see [doc/installation.md](doc/installation.md)

# Support
If you have any question, bug to report or would like to propose a new feature, feel free to create an [issue][issue] on GitHub.

[issue]: https://github.com/ricosjp/monolish/issues

# Contributing
If you want to contribute to monolish, create an [Issue][issue] and create a pull request for the Issue.
Before resolving the WIP of the pull request, you need to do the following two things

- Apply clang-format with `make format` command.
- Record the major fixes and the URL of the pull request in `CHANGELOG.md`.

The `make format` command will automatically apply clang-format to all git added files.

License
--------
Copyright 2021 RICOS Co. Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
