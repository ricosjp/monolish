cat double_blas_lv1.cpp \
    | sed -e 's/double/float/g' \
    | sed -e 's/dasum/sasum/g' \
    | sed -e 's/Dasum/Sasum/g' \
    | sed -e 's/daxpy/saxpy/g' \
    | sed -e 's/Daxpy/Saxpy/g' \
    | sed -e 's/ddot/sdot/g' \
    | sed -e 's/Ddot/Sdot/g' \
    | sed -e 's/dnrm2/snrm2/g' \
    | sed -e 's/Dnrm2/Snrm2/g' \
    | sed -e 's/dscal/sscal/g' \
    | sed -e 's/Dscal/Sscal/g' \
    > float_blas_lv1.cpp

cat double_addsub.cpp \
    | sed -e 's/double/float/g' \
    > float_addsub.cpp
