cat ./double_scalar_vector.cpp | sed -e 's/double/float/g'  > float_scalar_vector.cpp
cat ./double_vector_vector.cpp | sed -e 's/double/float/g'  > float_vector_vector.cpp
cat ./double_matrix_matrix.cpp | sed -e 's/double/float/g'  > float_matrix_matrix.cpp
cat ./double_scalar_matrix.cpp | sed -e 's/double/float/g'  > float_scalar_matrix.cpp
cat ./double_vector_math.cpp | sed -e 's/double/float/g'  > float_vector_math.cpp
