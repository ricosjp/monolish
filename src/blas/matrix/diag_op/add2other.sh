sed "s/add/sub/g" ./dense_diag_scalar_add.cpp | sed "s/+=/-=/g" > ./dense_diag_scalar_sub.cpp
sed "s/add/mul/g" ./dense_diag_scalar_add.cpp | sed "s/+=/*=/g" > ./dense_diag_scalar_mul.cpp
sed "s/add/div/g" ./dense_diag_scalar_add.cpp | sed "s/+=/\/=/g" > ./dense_diag_scalar_div.cpp

sed "s/add/sub/g" ./dense_diag_vector_add.cpp | sed "s/+=/-=/g" > ./dense_diag_vector_sub.cpp
sed "s/add/mul/g" ./dense_diag_vector_add.cpp | sed "s/+=/*=/g" > ./dense_diag_vector_mul.cpp
sed "s/add/div/g" ./dense_diag_vector_add.cpp | sed "s/+=/\/=/g" > ./dense_diag_vector_div.cpp
