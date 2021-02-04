#!/bin/bash
cat dense_double_getrs.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dgetrs/sgetrs/g' \
    | sed -e 's/Dgetrs/Sgetrs/g' \
    > dense_float_getrs.cpp
