sed 's/double precision/single precision/g' monolish_blas_double.hpp | \
    sed 's/double/float/g' > monolish_blas_float.hpp
sed 's/double precision/single precision/g' monolish_lapack_double.hpp | \
    sed 's/double/float/g' > monolish_lapack_float.hpp
