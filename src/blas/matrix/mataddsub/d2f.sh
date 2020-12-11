cat double_matadd.cpp \
    | sed -e 's/double/float/g' \
    > float_matadd.cpp
