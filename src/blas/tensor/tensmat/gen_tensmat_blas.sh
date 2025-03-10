#!/bin/bash

TRANSPOSE_BOOL(){
    if [ $1 = "N" ]
    then
        echo "false"
    else
        echo "true"
    fi
}

echo "//this code is generated by gen_tensmat_blas.sh
#include \"../../../../include/monolish_blas.hpp\"
#include \"../../../internal/monolish_internal.hpp\"
#include \"tensor_dense-dense_tensmat.hpp\"
#include \"tensor_crs-dense_tensmat.hpp\"

namespace monolish::blas {
"

## tensmat tensor_Dense-Dense
for prec in double float; do
  if [ $prec = "double" ]
  then
    for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void tensmat(const $arg1 &A, const $arg2 &B, $arg3 &C){tensor_Dense_Dense_Dtensmat_core(1.0, A, B, 0.0, C, false, false);}"
          echo "void tensmat(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){tensor_Dense_Dense_Dtensmat_core(a, A, B, b, C, false, false);}"
        done
      done
    done
  else
    for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void tensmat(const $arg1 &A, const $arg2 &B, $arg3 &C){tensor_Dense_Dense_Stensmat_core(1.0, A, B, 0.0, C, false, false);}"
          echo "void tensmat(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){tensor_Dense_Dense_Stensmat_core(a, A, B, b, C, false, false);}"
        done
      done
    done
  fi
done

echo ""

## tensmat tensor_CRS-Dense
for prec in double float; do
  if [ $prec = "double" ]
  then
    for arg1 in tensor::tensor_CRS\<$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void tensmat(const $arg1 &A, const $arg2 &B, $arg3 &C){tensor_CRS_Dense_Dtensmat_core(1.0, A, B, 0.0, C);}"
          echo "void tensmat(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){tensor_CRS_Dense_Dtensmat_core(a, A, B, b, C);}"
        done
      done
    done
  else
    for arg1 in tensor::tensor_CRS\<$prec\>; do
      for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
          echo "void tensmat(const $arg1 &A, const $arg2 &B, $arg3 &C){tensor_CRS_Dense_Stensmat_core(1.0, A, B, 0.0, C);}"
          echo "void tensmat(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){tensor_CRS_Dense_Stensmat_core(a, A, B, b, C);}"
        done
      done
    done
  fi
done

echo ""

echo "}"
