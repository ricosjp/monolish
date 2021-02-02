#!/bin/bash
cat dense_double_sytrf.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dsytrf/ssytrf/g' \
    | sed -e 's/Dsytrf/Ssytrf/g' \
    > dense_float_sytrf.cpp
