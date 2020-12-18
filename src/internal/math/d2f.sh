cat double_math.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/vd/vs/g' \
    > float_math.cpp
