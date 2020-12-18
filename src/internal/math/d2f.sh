cat double_math.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/vdTanh/vsTanh/g' \
    > float_math.cpp
