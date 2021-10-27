# monolish: MONOlithic LInear equation Solvers for Highly-parallel architecture. {#mainpage}

## Quick start guide
The GPU implementation requires OpenMP Offloading in clang. Enabling OpenMP Offloading requires an option to be specified when building clang. This is not practical.

To use pre-built monolish, use apt or Docker Container with monolish installed.
- [Installation using apt (for ubuntu 20.04)](@ref install_guide)
- [Docker container with monolish installed](@ref monolish_docker) ([container registry](https://github.com/orgs/ricosjp/packages?repo_name=monolish))

To share the development environment, the monolish development environment is provided in an [allgebra Docker container](https://github.com/ricosjp/allgebra).

See below for how to build on allgebra.
- [Build monolish from source code](@ref build_guide)

## For Users
- [What's monolish?](@ref monolish_intro)
- Installation
  - [Installation using apt (for ubuntu 20.04)](@ref install_guide)
  - [Docker container with monolish installed](@ref monolish_docker)
- [Data types](@ref data_type)
- [Compile and run simple program on CPU](@ref cpu_dev)
- [GPU device acceleration](@ref gpu_dev)
- [Matrix/vector operations](@ref oplist)
- [Linear solvers](@ref solverlist)
- [Performance logging and find bottlenecks](@ref logger)
- [View continuous benchmarking results](https://ricosjp.github.io/monolish_benchmark_result/)

## For Developers
- [Build monolish from source code](@ref build_guide)
- [Testing and Benchmarking](@ref test_bench)
- [Contribution guide](@ref contribution) 

## Citations
- [Publications](@ref publications)

## Links
- [Source code](https://github.com/ricosjp/monolish/)
- [Documentation](https://ricosjp.github.io/monolish/)
- [Bug reports](https://github.com/ricosjp/monolish/issues)
- [Releases](https://github.com/ricosjp/monolish/releases)
- [Changelog](https://github.com/ricosjp/monolish/blob/master/CHANGELOG.md)
- [Licence](https://github.com/ricosjp/monolish/blob/master/LICENSE)
- [monolish log viewer](https://pypi.org/project/monolish-log-viewer/)

Copyright 2021 [RICOS Co. Ltd.](https://www.ricos.co.jp/)
