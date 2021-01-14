#!/bin/bash
sed 's/double precision/single precision/g' monolish_lapack_double.hpp | \
    sed 's/double/float/g' > monolish_lapack_float.hpp
