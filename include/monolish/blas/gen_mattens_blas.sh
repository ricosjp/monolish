#!/bin/bash
echo "//this code is generated by gen_mattens_blas.sh
#pragma once
#include \"../common/monolish_common.hpp\"

namespace monolish {
/**
* @brief
* Basic Linear Algebra Subprograms for Dense Matrix, Sparse Matrix, Vector and
* Scalar
*/
namespace blas {
"

echo "
/**
 * @addtogroup BLASLV3
 * @{
 */
"

## mattens tensor_Dense
echo "
/**
 * \defgroup mattens_dense monolish::blas::mattens (tensor_Dense)
 * @brief matrix and tensor_Dense tensor multiplication: y = Ax
 * @{
 */
/**
 * @brief matrix and tensor_Dense tensor multiplication: ex. y_{ikl} = A_{ij} x_{jkl}
 * @param A Dense matrix
 * @param x tensor_Dense tensor
 * @param y tensor_Dense tensor
 * @note
 * - # of computation: ?
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void mattens(const $arg1 &A, const $arg2 &x, $arg3 &y);"
      done
    done
  done
done
echo "/**@}*/"

echo "
/**
 * \defgroup mattens_dense monolish::blas::mattens (tensor_Dense)
 * @brief matrix and tensor_Dense tensor multiplication: ex. y_{ikl} = a A_{ij} x_{jkl} + b y_{ikl}
 * @{
 */
/**
 * @brief matrix and tensor_Dense tensor multiplication: y = Ax
 * @param A Dense matrix
 * @param x tensor_Dense tensor 
 * @param y tensor_Dense tensor
 * @note
 * - # of computation: ?
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void mattens(const $prec &a, const $arg1 &A, const $arg2 &x, const $prec &b, $arg3 &y);"
      done
    done
  done
done
echo "/**@}*/"

echo "
/**
 * \defgroup mattens_dense monolish::blas::mattens (tensor_Dense)
 * @brief matrix and tensor_Dense tensor multiplication: y = Ax
 * @{
 */
/**
 * @brief matrix and tensor_Dense tensor multiplication: ex. y_{ikl} = A_{ij} x_{jkl}
 * @param A Dense matrix
 * @param x tensor_Dense tensor
 * @param y tensor_Dense tensor
 * @note
 * - # of computation: ?
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in matrix::CRS\<$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void mattens(const $arg1 &A, const $arg2 &x, $arg3 &y);"
      done
    done
  done
done
echo "/**@}*/"

echo "
/**
 * \defgroup mattens_dense monolish::blas::mattens (tensor_Dense)
 * @brief matrix and tensor_Dense tensor multiplication: ex. y_{ikl} = a A_{ij} x_{jkl} + b y_{ikl}
 * @{
 */
/**
 * @brief matrix and tensor_Dense tensor multiplication: y = Ax
 * @param A Dense matrix
 * @param x tensor_Dense tensor 
 * @param y tensor_Dense tensor
 * @note
 * - # of computation: ?
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in matrix::CRS\<$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void mattens(const $prec &a, const $arg1 &A, const $arg2 &x, const $prec &b, $arg3 &y);"
      done
    done
  done
done
echo "/**@}*/"



echo "/**@}*/"
echo "/**@}*/"
echo "}"
echo "}"
