GitLab CI settings
==================

[GitLab CI YAML reference](https://docs.gitlab.com/ee/ci/yaml/)

Stage
-----

### build

- [build.yml](./build.yml)
- [build_generated.yml](./build_generated.yml)

Responsible for compiling C++ code in `src/` into a shared library and for compiling `test/` codes.
This can be run in CPU runners since compiler does not requires GPU,
although running output executable requires GPU.
Generated library and test executables are passed to next stages as artifacts.
