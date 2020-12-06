cat double_arithmetic.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/vd/vs/g' \
    > float_arithmetic.cpp
