#!/bin/bash
echo "// This code is generated by gen_tensaddsub.sh
#include \"tensor_dense_tensaddsub.hpp\"
#include \"tensor_crs_tensaddsub.hpp\"

namespace monolish::blas {
"

# tensadd dense
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void tensadd(const $arg1 &A, const $arg2 &B, $arg3 &C) { tensadd_core(A, B, C); }"
      done
    done
  done
done

echo ""

# tensadd crs
for prec in double float; do
  echo "void tensadd(const tensor::tensor_CRS<$prec> &A, const tensor::tensor_CRS<$prec> &B, tensor::tensor_CRS<$prec> &C) { tensadd_core(A, B, C); }"
done

echo ""

# tenssub dense
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void tenssub(const $arg1 &A, const $arg2 &B, $arg3 &C) { tenssub_core(A, B, C); }"
      done
    done
  done
done

echo ""

# tenssub crs
for prec in double float; do
  echo "void tenssub(const tensor::tensor_CRS<$prec> &A, const tensor::tensor_CRS<$prec> &B, tensor::tensor_CRS<$prec> &C) { tenssub_core(A, B, C); }"
done

echo ""

echo "}"
