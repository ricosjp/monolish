cat double_arithmetic.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/vd/vs/g' \
    | sed -e 's/Dcopy/Scopy/g' \
    | sed -e 's/dcopy/scopy/g' \
    | sed -e 's/Dscal/Sscal/g' \
    | sed -e 's/dscal/sscal/g' \
    > float_arithmetic.cpp
