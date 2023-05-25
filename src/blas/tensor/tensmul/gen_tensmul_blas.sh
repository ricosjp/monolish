#!/bin/bash

TRANSPOSE_BOOL(){
    if [ $1 = "N" ]
    then
        echo "false"
    else
        echo "true"
    fi
}

echo "//this code is generated by gen_tensmul_blas.sh
#include \"../../../../include/monolish_blas.hpp\"
#include \"../../../internal/monolish_internal.hpp\"
#include \"tensor_dense-tensor_dense_tensmul.hpp\"

namespace monolish::blas {
"

## tensmul tensor_Dense-tensor_Dense
for prec in double float; do
  if [ $prec = "double" ]
  then
    for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void tensmul(const $arg1 &A, const $arg2 &B, $arg3 &C){tensor_Dense_tensor_Dense_Dtensmul_core(1.0, A, B, 0.0, C, false, false);}"
          echo "void tensmul(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){tensor_Dense_tensor_Dense_Dtensmul_core(a, A, B, b, C, false, false);}"
        done
      done
    done
  else
    for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void tensmul(const $arg1 &A, const $arg2 &B, $arg3 &C){tensor_Dense_tensor_Dense_Stensmul_core(1.0, A, B, 0.0, C, false, false);}"
          echo "void tensmul(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){tensor_Dense_tensor_Dense_Stensmul_core(a, A, B, b, C, false, false);}"
        done
      done
    done
  fi
done

echo ""

echo "}"
