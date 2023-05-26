#!/bin/bash

TRANSPOSE_BOOL(){
    if [ $1 = "N" ]
    then
        echo "false"
    else
        echo "true"
    fi
}

echo "//this code is generated by gen_mattens_blas.sh
#include \"../../../../include/monolish_blas.hpp\"
#include \"../../../internal/monolish_internal.hpp\"
#include \"crs-tensor_dense_mattens.hpp\"
#include \"dense-tensor_dense_mattens.hpp\"

namespace monolish::blas {
"

## mattens Dense-tensor_Dense
for prec in double float; do
  if [ $prec = "double" ]
  then
    for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void mattens(const $arg1 &A, const $arg2 &B, $arg3 &C){Dense_tensor_Dense_Dmattens_core(1.0, A, B, 0.0, C, false, false);}"
          echo "void mattens(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){Dense_tensor_Dense_Dmattens_core(a, A, B, b, C, false, false);}"
        done
      done
    done
  else
    for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void mattens(const $arg1 &A, const $arg2 &B, $arg3 &C){Dense_tensor_Dense_Smattens_core(1.0, A, B, 0.0, C, false, false);}"
          echo "void mattens(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){Dense_tensor_Dense_Smattens_core(a, A, B, b, C, false, false);}"
        done
      done
    done
  fi
done

echo ""
## mattens CRS-tensor_Dense
for prec in double float; do
  if [ $prec = "double" ]
  then
    for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void mattens(const matrix::CRS<$prec> &A, const $arg1 &B, $arg2 &C){CRS_tensor_Dense_Dmattens_core(1.0, A, B, 0.0, C);}"
        echo "void mattens(const $prec &a, const matrix::CRS<$prec> &A, const $arg1 &B, const $prec &b, $arg2 &C){CRS_tensor_Dense_Dmattens_core(a, A, B, b, C);}"
      done
    done
  else
    for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void mattens(const matrix::CRS<$prec> &A, const $arg1 &B, $arg2 &C){CRS_tensor_Dense_Smattens_core(1.0, A, B, 0.0, C);}"
        echo "void mattens(const $prec &a, const matrix::CRS<$prec> &A, const $arg1 &B, const $prec &b, $arg2 &C){CRS_tensor_Dense_Smattens_core(a, A, B, b, C);}"
      done
    done
  fi
done

echo ""

echo "}"
