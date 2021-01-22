#!/bin/bash
cat dense_double_sygv.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dsygvd/ssygvd/g' \
    | sed -e 's/Dsygvd/Ssygvd/g' \
    > dense_float_sygv.cpp
