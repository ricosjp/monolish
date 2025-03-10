#!/bin/bash
echo "//this code is generated by gen_matvec_blas.sh
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

## matrix Dense times_row
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Row-wise Dense matrix and vector times: 
 * C[i][j] = A[i][j] * x[j]
 * @param A Dense matrix (size M x N)
 * @param x monolish vector (size M)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
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
 * @brief Specified row of dense matrix and vector times: 
 * C[num][j] = A[num][j] * x[j]
 * @param A Dense matrix (size M x N)
 * @param num row number
 * @param x monolish vector (size M)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void times_row(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

## matrix CRS times_row
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Row-wise CRS matrix and vector times: 
 * C[i][j] = A[i][j] * x[j]
 * @param A CRS matrix (size M x N)
 * @param x monolish vector (size M)
 * @param C CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */ "
 for prec in double float; do
     for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::CRS\<$prec\>,$prec\>; do
         echo "void times_row(const matrix::CRS<$prec> &A, const $arg1 &x, matrix::CRS<$prec> &C);"
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
 * @brief Specified row of CRS matrix and vector times: 
 * C[num][j] = A[num][j] * x[j]
 * @param A CRS matrix (size M x N)
 * @param num row number
 * @param x monolish vector (size M)
 * @param C CRS matrix (size M x N)
 * @note
 * - # of computation: nnz of specified row
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */ "
 for prec in double float; do
     for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::CRS\<$prec\>,$prec\>; do
         echo "void times_row(const matrix::CRS<$prec> &A, const size_t num, const $arg1 &x, matrix::CRS<$prec> &C);"
     done
 done
 echo "/**@}*/"

## matrix Dense times_col
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Column-wise Dense matrix and vector times: 
 * C[i][j] = A[i][j] * x[i]
 * @param A Dense matrix (size M x N)
 * @param x monolish vector (size M)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
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
 * @brief Specified col of dense matrix and vector times: 
 * C[i][num] = A[i][num] * x[i]
 * @param A Dense matrix (size M x N)
 * @param num column number
 * @param x monolish vector (size M)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: M
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void times_col(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

## matrix CRS times_col
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
 /**
 * @brief Column-wise CRS matrix and vector times: 
 * C[i][j] = A[i][j] * x[i]
 * @param A CRS matrix (size M x N)
 * @param x monolish vector (size M)
 * @param C CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */ "
 for prec in double float; do
     for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::CRS\<$prec\>,$prec\>; do
         echo "void times_col(const matrix::CRS<$prec> &A, const $arg1 &x, matrix::CRS<$prec> &C);"
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
 * @brief Specified col of CRS matrix and vector times: 
 * C[i][num] = A[i][num] * x[i]
 * @param A CRS matrix (size M x N)
 * @param num column number
 * @param x monolish vector (size M)
 * @param C CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */ "
 for prec in double float; do
     for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::CRS\<$prec\>,$prec\>; do
         echo "void times_col(const matrix::CRS<$prec> &A, const size_t num, const $arg1 &x, matrix::CRS<$prec> &C);"
     done
 done
 echo "/**@}*/"

## matrix Dense adds_row
echo "
/**
 * \defgroup times monolish::blas::adds
 * @brief element by element addition
 * @{
 */
 /**
 * @brief Row-wise Dense matrix and vector adds: 
 * C[i][j] = A[i][j] + x[j]
 * @param A Dense matrix (size M x N)
 * @param x monolish vector (size M)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
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
 * @brief Specified row of dense matrix and vector adds: 
 * C[num][j] = A[num][j] + x[j]
 * @param A Dense matrix (size M x N)
 * @param num row number
 * @param x monolish vector (size M)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void adds_row(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

## matrix Dense adds_col
echo "
/**
 * \defgroup times monolish::blas::adds
 * @brief element by element addition
 * @{
 */
 /**
 * @brief Row-wise Dense matrix and vector adds: 
 * C[i][j] = A[i][j] + x[i]
 * @param A Dense matrix (size M x N)
 * @param x monolish vector (size M)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
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
 * @brief Specified col of dense matrix and vector adds: 
 * C[i][num] = A[i][num] + x[i]
 * @param A Dense matrix (size M x N)
 * @param num col number
 * @param x monolish vector (size M)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
 for prec in double float; do
   for arg1 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
     for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
       for arg3 in matrix::Dense\<$prec\> view_Dense\<vector\<$prec\>,$prec\> view_Dense\<matrix::Dense\<$prec\>,$prec\> view_Dense\<tensor::tensor_Dense\<$prec\>,$prec\>; do
         echo "void adds_col(const $arg1 &A, const size_t num, const $arg2 &x, $arg3 &C);"
       done
     done
   done
 done
 echo "/**@}*/"

## matvec Dense
echo "
/**
 * \defgroup matvec_dense monolish::blas::matvec (Dense)
 * @brief Dense matrix and vector multiplication: y = Ax
 * @{
 */
/**
 * @brief Dense matrix and vector multiplication: y = Ax
 * @param A Dense matrix (size M x N)
 * @param x monolish vector (size M)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void matvec(const matrix::Dense<$prec> &A, const $arg1 &x, $arg2 &y);"
    done
  done
done
echo "/**@}*/"

echo "
/**
 * \defgroup matvec_dense monolish::blas::matvec (Dense)
 * @brief Dense matrix and vector multiplication: y = aAx + by
 * @{
 */
/**
 * @brief Dense matrix and vector multiplication: y = aAx + by
 * @param A Dense matrix (size M x N)
 * @param x monolish vector (size M)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void matvec(const $prec &a, const matrix::Dense<$prec> &A, const $arg1 &x, const $prec &b, $arg2 &y);"
    done
  done
done
echo "/**@}*/"

## matvec_* Dense
for trans in N T; do
    echo "
    /**
    * \defgroup matvec_dense_$trans monolish::blas::matvec_$trans (Dense)
    * @brief Dense matrix and vector multiplication: y = A^$trans x
    * @{
    */
    /**
    * @brief Dense matrix and vector multiplication: y = A^$trans x
    * @param A Dense matrix (size M x N)
    * @param x monolish vector (size M)
    * @param y monolish vector (size M)
    * @note
    * - # of computation: MN
    * - Multi-threading: true
    * - GPU acceleration: true
    *    - # of data transfer: 0
    */ "
    for prec in double float; do
        for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
            for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
                echo "void matvec_$trans(const matrix::Dense<$prec> &A, const $arg1 &x, $arg2 &y);"
            done
        done
    done
echo "/**@}*/"
done

for trans in N T; do
    echo "
    /**
    * \defgroup matvec_dense_$trans monolish::blas::matvec_$trans (Dense)
    * @brief Dense matrix and vector multiplication: y = aA^$trans x + by
    * @{
    */
    /**
    * @brief Dense matrix and vector multiplication: y = aA^$trans x + by
    * @param A Dense matrix (size M x N)
    * @param x monolish vector (size M)
    * @param y monolish vector (size M)
    * @note
    * - # of computation: MN
    * - Multi-threading: true
    * - GPU acceleration: true
    *    - # of data transfer: 0
    */ "
    for prec in double float; do
        for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
            for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
                echo "void matvec_$trans(const $prec &a, const matrix::Dense<$prec> &A, const $arg1 &x, const $prec &b, $arg2 &y);"
            done
        done
    done
echo "/**@}*/"
done

## matvec CRS
echo "
/**
 * \defgroup matvec_crs monolish::blas::matvec (CRS)
 * @brief CRS format sparse matrix and vector multiplication: y = Ax
 * @{
 */
/**
 * @brief CRS format sparse matrix and vector multiplication: y = Ax
 * @param A CRS matrix (size M x N)
 * @param x monolish vector (size M)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: 2nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void matvec(const matrix::CRS<$prec> &A, const $arg1 &x, $arg2 &y);"
    done
  done
done
echo "/**@}*/"

echo "
/**
 * \defgroup matvec_crs monolish::blas::matvec (CRS)
 * @brief CRS format sparse matrix and vector multiplication: y = aAx + by
 * @{
 */
/**
 * @brief CRS format sparse matrix and vector multiplication: y = aAx + by
 * @param A CRS matrix (size M x N)
 * @param x monolish vector (size M)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: 2nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void matvec(const $prec &a, const matrix::CRS<$prec> &A, const $arg1 &x, const $prec &b, $arg2 &y);"
    done
  done
done
echo "/**@}*/"

## matvec_* CRS
for trans in N T; do
    echo "
    /**
    * \defgroup matvec_crs_$trans monolish::blas::matvec_$trans (CRS)
    * @brief CRS format sparse matrix and vector multiplication: y = A^$trans x
    * @{
    */
    /**
    * @brief CRS format sparse matrix and vector multiplication: y = A^$trans x
    * @param A CRS matrix (size M x N)
    * @param x monolish vector (size M)
    * @param y monolish vector (size M)
    * @note
    * - # of computation: 2nnz
    * - Multi-threading: true
    * - GPU acceleration: true
    *    - # of data transfer: 0
    */ "
    for prec in double float; do
        for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
            for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
                echo "void matvec_$trans(const matrix::CRS<$prec> &A, const $arg1 &x, $arg2 &y);"
            done
        done
    done
echo "/**@}*/"
done

for trans in N T; do
    echo "
    /**
    * \defgroup matvec_crs_$trans monolish::blas::matvec_$trans (CRS)
    * @brief CRS format sparse matrix and vector multiplication: y = aA^$trans x + by
    * @{
    */
    /**
    * @brief CRS format sparse matrix and vector multiplication: y = aA^$trans x + by
    * @param A CRS matrix (size M x N)
    * @param x monolish vector (size M)
    * @param y monolish vector (size M)
    * @note
    * - # of computation: 2nnz
    * - Multi-threading: true
    * - GPU acceleration: true
    *    - # of data transfer: 0
    */ "
    for prec in double float; do
        for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
            for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
                echo "void matvec_$trans(const $prec &a, const matrix::CRS<$prec> &A, const $arg1 &x, const $prec &b, $arg2 &y);"
            done
        done
    done
echo "/**@}*/"
done


## matvec LinearOperator
echo "
/**
 * \defgroup matvec_LO monolish::blas::matvec (LinearOperator)
 * @brief LinearOperator matrix and vector multiplication: y = Ax
 * @{
 */
/**
 * @brief matrix (LinearOperator) and vector multiplication: y = Ax
 * @param A LinearOperator (size M x N)
 * @param x monolish vector (size N)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: depends on matvec function
 * - Multi-threading: depends on matvec function
 * - GPU acceleration: depends on matvec function
 */ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void matvec(const matrix::LinearOperator<$prec> &A, const $arg1 &x, $arg2 &y);"
    done
  done
done

echo "/**@}*/"

## rmatvec LinearOperator
echo "
/**
 * \defgroup rmatvec_LO monolish::blas::rmatvec (LinearOperator)
 * @brief Adjoint LinearOperator matrix and vector multiplication: y = A^Hx
 * @{
 */
/**
 * @brief Adjoint LinearOperator matrix and vector multiplication: y = A^Hx
 * @param A LinearOperator (size M x N)
 * @param x monolish vector (size N)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: depends on matvec function
 * - Multi-threading: depends on matvec function
 * - GPU acceleration: depends on matvec function
 */ "
for prec in double float; do
  for arg1 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
    for arg2 in vector\<$prec\> view1D\<vector\<$prec\>,$prec\> view1D\<matrix::Dense\<$prec\>,$prec\> view1D\<tensor::tensor_Dense\<$prec\>,$prec\>; do
      echo "void rmatvec(const matrix::LinearOperator<$prec> &A, const $arg1 &x, $arg2 &y);"
    done
  done
done

echo "/**@}*/"
echo "/**@}*/"
echo "}"
echo "}"
