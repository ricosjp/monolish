sed "s/add/sub/g" ./tensor_dense_diag_add.cpp | sed "s/+=/-=/g" > ./tensor_dense_diag_sub.cpp
sed "s/add/mul/g" ./tensor_dense_diag_add.cpp | sed "s/+=/*=/g" > ./tensor_dense_diag_mul.cpp
sed "s/add/div/g" ./tensor_dense_diag_add.cpp | sed "s/+=/\/=/g" > ./tensor_dense_diag_div.cpp
