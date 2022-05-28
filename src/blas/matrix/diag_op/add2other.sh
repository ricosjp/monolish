sed "s/add/sub/g" ./dense_diag_add.cpp | sed "s/+=/-=/g" > ./dense_diag_sub.cpp
sed "s/add/mul/g" ./dense_diag_add.cpp | sed "s/+=/*=/g" > ./dense_diag_mul.cpp
sed "s/add/div/g" ./dense_diag_add.cpp | sed "s/+=/\/=/g" > ./dense_diag_div.cpp
