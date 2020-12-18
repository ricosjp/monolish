cat double_arithmetic.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/vdAdd/vsAdd/g' \
    | sed -e 's/vdSub/vsSub/g' \
    | sed -e 's/vdMul/vsMul/g' \
    | sed -e 's/vdDiv/vsDiv/g' \
    | sed -e 's/Dcopy/Scopy/g' \
    | sed -e 's/dcopy/scopy/g' \
    > float_arithmetic.cpp
