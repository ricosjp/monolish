# Performance logging and find bottlenecks {#logger}
# Enable performance logging
monolish provides a logger to measure the execution time of each function at runtime and find bottlenecks.
Functions on elements cannot be logged because the processing time is too short.

To enable logging, call the following function.

```
monolish::util::set_log_level(3);
monolish::util::set_log_filename(filename);
```

The default log level is zero, so normally the logger does not work.

By changing the log level, the detail of the log is determined.
```
0: No logging
1: Linear solver only
2: Linear solver and BLAS functions
3: Linear solver, BLAS functions, and Util functions
```

`set_log_filename` specifies the name of the output file.
If the output file name is not specified, it will be output to the standard output.

The output log is in yml format.

## Install monolish-log-viewer
The monolish project has developed a tool to analyze and visualize the output log files.
The program is available in `monolish/python/`.

It has been uploaded to [PyPI](https://pypi.org/project/monolish-log-viewer/) sever and can be installed using the `pip` command.

If pip is not available, use the following command.
```
apt install python-3-pip
```

Install the logger using pip. Execute the following command:
```
python3 -m pip install monolish-log-viewer
```

## How to use monolish-log-viewer
`monolish_log_viewer` outputs HTML files from log files in yml format output by monolish.
To generate log.html from log.yml, execute the following command.

```
monolish_log_viewer log.yml log.html
```

A sample can be found below:
- [CPU log data](https://ricos.pages.ritc.jp/monolish/monolish_test_cpu.html)
- [GPU log data](https://ricos.pages.ritc.jp/monolish/monolish_test_gpu.html)
