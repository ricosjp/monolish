# monolish: MONOlithic LIner equation Solvers for Highly-parallel architecture
monolish is a linear equation solver library that monolithically fuses variable data type, matrix structures, matrix data format, vendor specific data transfer APIs, and vendor specific numerical algebra libraries.

- Documentation: https://ricosjp.github.io/monolish/
- Source code: https://github.com/ricosjp/monolish/
- Contribution guide : https://ricosjp.github.io/monolish/da/dce/contribution.html
- Publications: https://ricosjp.github.io/monolish/de/d26/publications.html
- Bug reports: https://github.com/ricosjp/monolish/issues
- Releases: https://github.com/ricosjp/monolish/releases
- Changelog: https://github.com/ricosjp/monolish/blob/master/CHANGELOG.md
- monolish log viewer: https://pypi.org/project/monolish-log-viewer/
- monolish benchmark result: https://ricosjp.github.io/monolish_benchmark_result/

---

monolish let developer forget about:
- Performance tuning
- Processor differences which execute library (Intel CPU, NVIDIA GPU, AMD CPU, ARM CPU, NEC SX-Aurora TSUBASA, etc.)
- Vendor specific data transfer APIs (host RAM to Device RAM)
- Finding bottlenecks and performance benchmarks
- The argument data type of matrix/vector operations
- Matrix structures and storage formats
- Cumbersome package dependency

Installation
---
The GPU implementation requires OpenMP Offloading in clang. Enabling OpenMP Offloading requires an option to be specified when building clang. This is not practical.

To use pre-built monolish, use apt or Docker Container with monolish installed.
- [Installation using apt (for ubuntu 20.04 and 20.10)](https://ricosjp.github.io/monolish/d5/d38/install_guide.html)
- [Docker container with monolish installed](https://ricosjp.github.io/monolish/d6/d6f/monolish_docker.html) ([container registory](https://github.com/orgs/ricosjp/packages?repo_name=monolish))

To share the development environment, the monolish development environment is provided in an [allgebra Docker container](https://github.com/ricosjp/allgebra).

See below for how to build on algebra.
- [Build monolish from source code](https://ricosjp.github.io/monolish/da/d20/build_guide.html)

License
--------
Copyright 2021 [RICOS Co. Ltd.](https://www.ricos.co.jp/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
