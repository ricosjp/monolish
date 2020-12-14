cat double_mataddsub.cpp \
    | sed -e 's/double/float/g' \
    > float_mataddsub.cpp
