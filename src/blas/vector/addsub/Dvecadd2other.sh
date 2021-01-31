cat double_vecadd.cpp \
    | sed -e 's/double/float/g' \
    > float_vecadd.cpp

cat double_vecadd.cpp \
    | sed -e 's/vecadd/vecsub/g' \
    | sed -e 's/vadd/vsub/g' \
    > double_vecsub.cpp

cat double_vecadd.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/vecadd/vecsub/g' \
    | sed -e 's/vadd/vsub/g' \
    > float_vecsub.cpp
