cat dense_double_syev.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dsyev/ssyev/g' \
    > dense_float_syev.cpp
