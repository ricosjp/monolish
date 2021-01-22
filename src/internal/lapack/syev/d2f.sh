#!/bin/bash
cat dense_double_syev.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dsyevd/ssyevd/g' \
    | sed -e 's/Dsyevd/Ssyevd/g' \
    > dense_float_syev.cpp
