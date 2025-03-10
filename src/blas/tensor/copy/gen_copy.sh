#!/bin/bash
echo "// This code is generated by gen_copy.sh
#include \"tensor_dense_copy.hpp\"

namespace monolish::blas {
"

# copy dense
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void copy(const $arg1 &A, $arg2 &C){ copy_core(A, C); }"
    done
  done
done

echo ""

# copy crs
for prec in double float; do
  echo "void copy(const tensor::tensor_CRS<$prec> &A, tensor::tensor_CRS<$prec> &C){ copy_core(A, C); }"
done

echo "}"
