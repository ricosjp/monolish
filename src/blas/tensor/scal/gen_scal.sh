#!/bin/bash
echo "// This code is generated by gen_scal.sh
#include \"tensor_dense_scal.hpp\"

namespace monolish::blas {
"

# scal dense
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    echo "void tscal(const $prec alpha, $arg1& A) { tscal_core(alpha, A); }"
  done
done

echo ""

# scal crs
for prec in double float; do
  echo "void mscal(const $prec alpha, tensor::tensor_CRS<$prec>& A) { mscal_core(alpha, A); }"
done

echo "}"
