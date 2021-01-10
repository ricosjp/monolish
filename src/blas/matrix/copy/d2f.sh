cat double_copy.cpp \
    | sed -e 's/double/float/g' \
    > float_copy.cpp
