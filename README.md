# monolish: MONOlithic LInear equation Solvers for Highly-parallel architecture

monolish is a linear equation solver library that monolithically fuses variable data type, matrix structures, matrix data format, vendor specific data transfer APIs, and vendor specific numerical algebra libraries.

monolish let developer forget about:

- Performance tuning
- Processor differences which execute library (Intel CPU, NVIDIA GPU, AMD CPU, ARM CPU, NEC SX-Aurora TSUBASA, etc.)
- Vendor specific data transfer APIs (host RAM to Device RAM)
- Finding bottlenecks and performance benchmarks
- The argument data type of matrix/vector operations
- Matrix structures and storage formats
- Cumbersome package dependency

Documents
----------

| monolish version | Release Note | Document |
|:-----------------|:-------------|:---------|
| master | [![badge](https://img.shields.io/badge/CHANGELOG-unreleased-yellow)](https://github.com/ricosjp/monolish/blob/master/CHANGELOG.md#unreleased) | [![badge](https://img.shields.io/badge/Document-master-blue)](https://ricosjp.github.io/monolish/master/) |
| 0.16.3 | [![badge](https://img.shields.io/badge/Release-0.16.3-green)](https://github.com/ricosjp/monolish/releases/tag/0.16.3) | [![badge](https://img.shields.io/badge/Document-0.16.3-blue)](https://ricosjp.github.io/monolish/0.16.3/) |
| 0.16.2 | [![badge](https://img.shields.io/badge/Release-0.16.2-green)](https://github.com/ricosjp/monolish/releases/tag/0.16.2) | [![badge](https://img.shields.io/badge/Document-0.16.2-blue)](https://ricosjp.github.io/monolish/0.16.2/) |
| 0.16.1 | [![badge](https://img.shields.io/badge/Release-0.16.1-green)](https://github.com/ricosjp/monolish/releases/tag/0.16.1) | [![badge](https://img.shields.io/badge/Document-0.16.1-blue)](https://ricosjp.github.io/monolish/0.16.1/) |
| 0.16.0 | [![badge](https://img.shields.io/badge/Release-0.16.0-green)](https://github.com/ricosjp/monolish/releases/tag/0.16.0) | [![badge](https://img.shields.io/badge/Document-0.16.0-blue)](https://ricosjp.github.io/monolish/0.16.0/) |
| 0.15.3 | [![badge](https://img.shields.io/badge/Release-0.15.3-green)](https://github.com/ricosjp/monolish/releases/tag/0.15.3) | [![badge](https://img.shields.io/badge/Document-0.15.3-blue)](https://ricosjp.github.io/monolish/0.15.3/) |
| 0.15.2 | [![badge](https://img.shields.io/badge/Release-0.15.2-green)](https://github.com/ricosjp/monolish/releases/tag/0.15.2) | [![badge](https://img.shields.io/badge/Document-0.15.2-blue)](https://ricosjp.github.io/monolish/0.15.2/) |
| 0.15.1 | [![badge](https://img.shields.io/badge/Release-0.15.1-green)](https://github.com/ricosjp/monolish/releases/tag/0.15.1) | [![badge](https://img.shields.io/badge/Document-0.15.1-blue)](https://ricosjp.github.io/monolish/0.15.1/) |
| 0.15.0 | [![badge](https://img.shields.io/badge/Release-0.15.0-green)](https://github.com/ricosjp/monolish/releases/tag/0.15.0) | [![badge](https://img.shields.io/badge/Document-0.15.0-blue)](https://ricosjp.github.io/monolish/0.15.0/) |
| 0.14.2 | [![badge](https://img.shields.io/badge/Release-0.14.2-green)](https://github.com/ricosjp/monolish/releases/tag/0.14.2) | [![badge](https://img.shields.io/badge/Document-0.14.2-blue)](https://ricosjp.github.io/monolish/0.14.2/) |
| 0.14.1 | [![badge](https://img.shields.io/badge/Release-0.14.1-green)](https://github.com/ricosjp/monolish/releases/tag/0.14.1) | [![badge](https://img.shields.io/badge/Document-0.14.1-blue)](https://ricosjp.github.io/monolish/0.14.1/) |
| 0.14.0 | [![badge](https://img.shields.io/badge/Release-0.14.0-green)](https://github.com/ricosjp/monolish/releases/tag/0.14.0) | [![badge](https://img.shields.io/badge/Document-0.14.0-blue)](https://ricosjp.github.io/monolish/0.14.0/) |

Links
-----

- Source code: <https://github.com/ricosjp/monolish/>
- Bug reports: <https://github.com/ricosjp/monolish/issues>
- Releases: <https://github.com/ricosjp/monolish/releases>
- monolish log viewer: <https://pypi.org/project/monolish-log-viewer/>
- monolish benchmark result: <https://ricosjp.github.io/monolish_benchmark_result/>

External projects
---

- [gomalish](https://github.com/AtelierArith/gomalish) : Julia interface of monolish
- [haskell-monolish](https://github.com/lotz84/haskell-monolish) : Haskell interface of monolish

License
--------

Copyright 2021 [RICOS Co. Ltd.](https://www.ricos.co.jp/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
