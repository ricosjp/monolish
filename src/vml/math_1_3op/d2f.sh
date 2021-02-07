cat ./double_matrix_math.cpp | sed -e 's/double/float/g'  > float_matrix_math.cpp
cat ./double_vector_math.cpp | sed -e 's/double/float/g'  > float_vector_math.cpp
