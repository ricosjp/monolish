#!/bin/bash
cat dense_double_getrf.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dgetrf/sgetrf/g' \
    | sed -e 's/Dgetrf/Sgetrf/g' \
    > dense_float_getrf.cpp
