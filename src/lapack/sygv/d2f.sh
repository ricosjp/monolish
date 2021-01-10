cat dense_double_sygv.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dsygv/ssygv/g' \
    > dense_float_sygv.cpp
