#!/bin/bash

TRANSPOSE_BOOL(){
    if [ $1 = "N" ]
    then
        echo "false"
    else
        echo "true"
    fi
}

echo "//this code is generated by gen_matvec_blas.sh
#include \"../../../../include/monolish_blas.hpp\"
#include \"../../../internal/monolish_internal.hpp\"
#include \"dense-dense_matmul.hpp\"
#include \"crs-dense_matmul.hpp\"
#include \"linearoperator-dense_matmul.hpp\"
#include \"linearoperator-linearoperator_matmul.hpp\"

namespace monolish::blas {
"

## matmul Dense-Dense
for prec in double float; do
  for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        if [ $prec = "double" ]
        then
            echo "void matmul(const $arg1 &A, const $arg2 &B, $arg3 &C){Dense_Dense_Dmatmul_core(1.0, A, B, 0.0, C, false, false);}"
            echo "void matmul(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){Dense_Dense_Dmatmul_core(a, A, B, b, C, false, false);}"
        else
            echo "void matmul(const $arg1 &A, const $arg2 &B, $arg3 &C){Dense_Dense_Smatmul_core(1.0, A, B, 0.0, C, false, false);}"
            echo "void matmul(const $prec &a, const $arg1 &A, const $arg2 &B, const $prec &b, $arg3 &C){Dense_Dense_Smatmul_core(a, A, B, b, C, false, false);}"
        fi
      done
    done
  done
done

echo ""

## matmul CRS-Dense
for prec in double float; do
  for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      if [ $prec = "double" ]
      then
          echo "void matmul(const matrix::CRS<$prec> &A, const $arg1 &B, $arg2 &C){CRS_Dense_Dmatmul_core(1.0, A, B, 0.0, C);}"
          echo "void matmul(const $prec &a, const matrix::CRS<$prec> &A, const $arg1 &B, const $prec &b, $arg2 &C){CRS_Dense_Dmatmul_core(a, A, B, b, C);}"
      else
          echo "void matmul(const matrix::CRS<$prec> &A, const $arg1 &B, $arg2 &C){CRS_Dense_Smatmul_core(1.0, A, B, 0.0, C);}"
          echo "void matmul(const $prec &a, const matrix::CRS<$prec> &A, const $arg1 &B, const $prec &b, $arg2 &C){CRS_Dense_Smatmul_core(a, A, B, b, C);}"
      fi
    done
  done
done

echo ""

# ## matmul_* Dense
# for transA in N T; do
#     for transB in N T; do
#         for prec in double float; do
#             if [ $prec = "double" ]
#             then
#                 TA=`TRANSPOSE_BOOL $transA`
#                 TB=`TRANSPOSE_BOOL $transB`
#                 echo "void matmul_$transA$transB(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C){Dense_Dense_Dmatmul_core(A, B, C, $TA, $TB);}"
#             else
#                 TA=`TRANSPOSE_BOOL $transA`
#                 TB=`TRANSPOSE_BOOL $transB`
#                 echo "void matmul_$transA$transB(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C){Dense_Dense_Smatmul_core(A, B, C, $TA, $TB);}"
#             fi
#         done
#     done
# done
# 
# echo ""


echo "}"
