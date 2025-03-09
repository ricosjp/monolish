#!/bin/bash
echo "//this code is generated by gen_tensvec_blas.sh
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
 * @addtogroup BLASLV2
 * @{
 */
"

## tensor tensor_Dense times_row
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Row-wise tensor_Dense tensor and vector times: 
 * ex. C[i][j] = A[i][j] * x[j]
 * @param A tensor_Dense tensor
 * @param x monolish vector
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: size
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void times_row(const $arg1 &A, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Specified row of tensor_Dense tensor and vector times: 
 * ex. C[num][j] = A[num][j] * x[j]
 * @param A tensor_Dense tensor
 * @param num row number (size N)
 * @param x monolish vector
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void times_row(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

## tensor tensor_CRS times_row
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Row-wise tensor_Dense tensor and vector times: 
 * ex. C[i][j] = A[i][j] * x[j]
 * @param A tensor_Dense tensor
 * @param x monolish vector
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: size
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     echo "void times_row(const tensor::tensor_CRS<$prec> &A, const $arg1 &x, tensor::tensor_CRS<$prec> &C);"
   done
 done
 echo "/**@}*/"

## tensor tensor_Dense times_col
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Column-wise tensor_Dense tensor and vector times: 
 * ex. C[i][j] = A[i][j] * x[i]
 * @param A tensor_Dense tensor
 * @param x monolish vector
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: size
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void times_col(const $arg1 &A, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Specified col of tensor_Dense tensor and vector times: 
 * C[i][num] = A[i][num] * x[i]
 * @param A tensor_Dense tensor
 * @param num column number
 * @param x monolish vector (size M)
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: M
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void times_col(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"


## tensor tensor_Dense adds_row
echo "
/**
 * \defgroup times monolish::blas::adds
 * @brief element by element addition
 * @{
 */
 /**
 * @brief Row-wise tensor_Dense tensor and vector adds: 
 * ex. C[i][j] = A[i][j] + x[j]
 * @param A tensor_Dense tensor
 * @param x monolish vector
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: size
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void adds_row(const $arg1 &A, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

echo "
/**
 * \defgroup times monolish::blas::adds
 * @brief element by element addition
 * @{
 */
 /**
 * @brief Specified row of tensor_Dense tensor and vector adds: 
 * ex. C[num][j] = A[num][j] + x[j]
 * @param A tensor_Dense tensor
 * @param num row number (size N)
 * @param x monolish vector
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void adds_row(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

## tensor tensor_Dense adds_col
echo "
/**
 * \defgroup times monolish::blas::adds
 * @brief element by element addition
 * @{
 */
 /**
 * @brief Column-wise tensor_Dense tensor and vector adds: 
 * ex. C[i][j] = A[i][j] + x[i]
 * @param A tensor_Dense tensor
 * @param x monolish vector
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: size
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void adds_col(const $arg1 &A, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

echo "
/**
 * \defgroup times monolish::blas::adds
 * @brief element by element addition
 * @{
 */
 /**
 * @brief Specified col of tensor_Dense tensor and vector adds: 
 * C[i][num] = A[i][num] + x[i]
 * @param A tensor_Dense tensor
 * @param num column number
 * @param x monolish vector (size M)
 * @param C tensor_Dense tensor
 * @note
 * - # of computation: M
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void adds_col(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

## tensvec tensor_Dense
echo "
/**
 * \defgroup tensvec_dense monolish::blas::tensvec (tensor_Dense)
 * @brief tensor_Dense tensor and vector multiplication: y = Ax
 * @{
 */
/**
 * @brief tensor_Dense tensor and vector multiplication: ex. y_{ij} = A_{ijk} x_{k}
 * @param A tensor_Dense tensor 
 * @param x monolish vector
 * @param y tensor_Dense tensor
 * @note
 * - # of computation: size
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      for arg3 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
        echo "void tensvec(const $arg1 &A, const $arg2 &x, $arg3 &y);"
      done
    done
  done
done
echo "/**@}*/"

## tensvec tensor_CRS
echo "
/**
 * \defgroup tensvec_dense monolish::blas::tensvec (tensor_CRS)
 * @brief tensor_Dense tensor and vector multiplication: y = Ax
 * @{
 */
/**
 * @brief tensor_Dense tensor and vector multiplication: ex. y_{ij} = A_{ijk} x_{k}
 * @param A tensor_Dense tensor 
 * @param x monolish vector
 * @param y tensor_Dense tensor
 * @note
 * - # of computation: size
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in tensor::tensor_Dense\<$prec\> view_tensor_Dense\<vector\<$prec\>,$prec\> view_tensor_Dense\<matrix::Dense\<$prec\>,$prec\> view_tensor_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void tensvec(const tensor::tensor_CRS<$prec> &A, const $arg1 &x, $arg2 &y);"
    done
  done
done
echo "/**@}*/"

echo "/**@}*/"
echo "/**@}*/"
echo "}"
echo "}"
