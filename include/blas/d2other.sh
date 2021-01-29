sed 's/double precision/single precision/g' monolish_blas_double_vector.hpp | \
    sed 's/double/float/g' > monolish_blas_float_vector.hpp

sed 's/double precision/single precision/g' monolish_blas_double_matrix_vector.hpp | \
    sed 's/double/float/g' > monolish_blas_float_matrix_vector.hpp

sed 's/double precision/single precision/g' monolish_blas_double_matrix.hpp | \
    sed 's/double/float/g' > monolish_blas_float_matrix.hpp



sed 's/vector<double>/view1D<vector<double>, double>/g' monolish_blas_double_vector.hpp | \
  sed 's/monolish vector/monolish view1D/g'> monolish_blas_double_vector_view1D-vector.hpp

sed 's/vector<double>/view1D<matrix::Dense<double>, double>/g' monolish_blas_double_vector.hpp | \
  sed 's/monolish vector/monolish view1D/g' > monolish_blas_double_vector_view1D-Dense.hpp

sed 's/vector<double>/view1D<vector<float>, float>/g' monolish_blas_double_vector.hpp | \
  sed 's/monolish vector/monolish view1D/g' > monolish_blas_float_vector_view1D-vector.hpp

sed 's/vector<double>/view1D<matrix::Dense<float>, float>/g' monolish_blas_double_vector.hpp | \
  sed 's/monolish vector/monolish view1D/g' > monolish_blas_float_vector_view1D-Dense.hpp



sed 's/vector<double>/view1D<vector<double>, double>/g' monolish_blas_double_vector.hpp | \
  sed 's/monolish vector/monolish view1D/g'> monolish_blas_double_matrix_view1D-vector.hpp

sed 's/vector<double>/view1D<matrix::Dense<double>, double>/g' monolish_blas_double_vector.hpp | \
  sed 's/monolish vector/monolish view1D/g' > monolish_blas_double_matrix_view1D-Dense.hpp

sed 's/vector<double>/view1D<vector<float>, float>/g' monolish_blas_double_vector.hpp | \
  sed 's/monolish vector/monolish view1D/g' > monolish_blas_float_matrix_view1D-vector.hpp

sed 's/vector<double>/view1D<matrix::Dense<float>, float>/g' monolish_blas_double_vector.hpp | \
  sed 's/monolish vector/monolish view1D/g' > monolish_blas_float_matrix_view1D-Dense.hpp
