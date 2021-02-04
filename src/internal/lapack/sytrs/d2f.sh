#!/bin/bash
cat dense_double_sytrs.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dsytrs/ssytrs/g' \
    | sed -e 's/Dsytrs/Ssytrs/g' \
    > dense_float_sytrs.cpp
