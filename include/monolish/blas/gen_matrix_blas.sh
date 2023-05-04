#!/bin/bash
echo "//this code is generated by gen_matrix_blas.sh
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

## copy Dense
echo "
/**
 * \defgroup mat_copy_Dense monolish::blas::copy (Dense)
 * @brief Dense matrix copy (C=A)
 * @{
 */
 /**
 * @brief Dense matrix copy (C=A)
 * @param A monolish Dense matrix (size M x N)
 * @param C monolish Dense matrix (size M x N)
 * @note
 * - # of computation: M x N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */ "
for prec in double float; do
  echo "void copy(const matrix::Dense<$prec> &A, matrix::Dense<$prec> &C);"
done

echo "/**@}*/"

## copy CRS
echo "
/**
 * \defgroup mat_copy_crs monolish::blas::copy (CRS)
 * @brief CRS matrix copy (y=a)
 * @{
 */
/**
 * @brief CRS matrix copy (y=a)
 * @param A monolish CRS matrix (size M x N)
 * @param C monolish CRS matrix (size M x N)
 * @note
 * - # of computation: M x N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */ "
for prec in double float; do
  echo "void copy(const matrix::CRS<$prec> &A, matrix::CRS<$prec> &C);"
done

echo "/**@}*/"

## copy LinearOperator
echo "
/**
 * \defgroup mat_copy_LO monolish::blas::copy (LinearOperator)
 * @brief LinearOperator copy (C=A)
 * @{
 */
/**
 * @brief LinearOperator copy (C=A)
 * @param A monolish LinearOperator (size M x N)
 * @param C monolish LinearOperator (size M x N)
 * @note
 * - # of computation: M x N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void copy(const matrix::LinearOperator<$prec> &A, matrix::LinearOperator<$prec> &C);"
done

echo "/**@}*/"

##############################################

#mscal Dense
echo "
/**
 * \defgroup mscal_dense monolish::blas::mscal (Dense)
 * @brief Dense matrix scal: A = alpha * A
 * @{
 */
/**
 * @brief Dense matrix scal: A = alpha * A
 * @param alpha scalar value
 * @param A Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void mscal(const $prec alpha, matrix::Dense<$prec> &A);"
done

echo "/**@}*/"

#mscal CRS
echo "
/**
 * \defgroup mscal_crs monolish::blas::mscal (CRS)
 * @brief CRS matrix scal: A = alpha * A
 * @{
 */
/**
 * @brief CRS matrix scal: A = alpha * A
 * @param alpha scalar value
 * @param A CRS matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void mscal(const $prec alpha, matrix::CRS<$prec> &A);"
done

echo "/**@}*/"

##############################################
# times scalar (almost same as mscal)

# times scalar Dense
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
/**
 * @brief Dense matrix times: C = alpha * A
 * @param alpha scalar value
 * @param A Dense matrix (size M x N)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void times(const $prec alpha, const matrix::Dense<$prec> &A, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

# times scalar CRS
echo "
/**
 * \defgroup times monolish::blas::times
 * @brief element by element multiplication
 * @{
 */
/**
 * @brief CRS matrix times: C = alpha * A
 * @param alpha scalar value
 * @param A CRS matrix (size M x N)
 * @param C CRS matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void times(const $prec alpha, const matrix::CRS<$prec> &A, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"

##############################################
# adds scalar

# adds scalar Dense
echo "
/**
 * \defgroup adds monolish::blas::adds
 * @brief element by element multiplication
 * @{
 */
/**
 * @brief Dense matrix adds: C = alpha + A
 * @param alpha scalar value
 * @param A Dense matrix (size M x N)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void adds(const $prec alpha, const matrix::Dense<$prec> &A, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

##############################################

#matadd Dense
echo "
/**
 * \defgroup madd_dense monolish::blas::matadd (Dense)
 * @brief Dense matrix addition: C = A + B
 * @{
 */
/**
 * @brief Dense matrix addition: C = A + B
 * @param A Dense matrix (size M x N)
 * @param B Dense matrix (size M x N)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void matadd(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

#matadd LinearOperator
echo "
/**
 * \defgroup madd_LO monolish::blas::matadd (LinearOperator)
 * @brief LinearOperator matrix addition: C = A + B
 * @{
 */
/**
 * @brief LinearOperator matrix addition: C = A + B
 * @param A LinearOperator (size M x N)
 * @param B LinearOperator (size M x N)
 * @param C LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void matadd(const matrix::LinearOperator<$prec> &A, const matrix::LinearOperator<$prec> &B, matrix::LinearOperator<$prec> &C);"
done
echo "/**@}*/"

#matadd CRS
echo "
/**
 * \defgroup madd_crs monolish::blas::matadd (CRS)
 * @brief CRS matrix addition: C = A + B 
 * @{
 */
/**
 * @brief CRS matrix addition: C = A + B
 * @param A CRS matrix (size M x N)
 * @param B CRS matrix (size M x N)
 * @param C CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void matadd(const matrix::CRS<$prec> &A, const matrix::CRS<$prec> &B, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"

echo ""

#matsub Dense
echo "
/**
 * \defgroup msub_dense monolish::blas::matsub (Dense)
 * @brief Dense matrix subtract: C = A - B
 * @{
 */
/**
 * @brief Dense matrix subtract: C = A - B
 * @param A Dense matrix (size M x N)
 * @param B Dense matrix (size M x N)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void matsub(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

#matsub LinearOperator
echo "
/**
 * \defgroup msub_LO monolish::blas::matsub (LinearOperator)
 * @brief LinearOperator subtract: C = A - B
 * @{
 */
/**
 * @brief LinearOperator subtract: C = A - B
 * @param A LinearOperator (size M x N)
 * @param B LinearOperator (size M x N)
 * @param C LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void matsub(const matrix::LinearOperator<$prec> &A, const matrix::LinearOperator<$prec> &B, matrix::LinearOperator<$prec> &C);"
done
echo "/**@}*/"

#matsub CRS
echo "
/**
 * \defgroup msub_crs monolish::blas::matsub (CRS)
 * @brief CRS matrix subtract: C = A - B (A and B must be
 * @{
 */
/**
 * @brief CRS matrix subtract: C = A - B (A and B must be
 * same non-zero structure)
 * @param A CRS matrix (size M x N)
 * @param B CRS matrix (size M x N)
 * @param C CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void matsub(const matrix::CRS<$prec> &A, const matrix::CRS<$prec> &B, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"

echo ""
#################################

#matmul Dense
echo "
/**
 * \defgroup mm_dense monolish::blas::matmul (Dense, Dense, Dense)
 * @brief Dense matrix multiplication: C = AB
 * @{
 */
/**
 * @brief Dense matrix multiplication: C = AB
 * @param A Dense matrix (size M x K)
 * @param B Dense matrix (size K x N)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: 2MNK
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void matmul(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

echo "
/**
 * \defgroup mm_dense monolish::blas::matmul (Float, Dense, Dense, Float, Dense)
 * @brief Dense matrix multiplication: C = aAB+bC
 * @{
 */
/**
 * @brief Dense matrix multiplication: C = aAB+bC
 * @param a Float
 * @param A Dense matrix (size M x K)
 * @param B Dense matrix (size K x N)
 * @param b Float
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: 2MNK
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void matmul(const $prec &a, const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, const $prec &b, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

#matmul_* Dense
# for TA in N T; do
# for TB in N T; do
# echo "
# /**
#  * \defgroup mm_dense_$TA$TB monolish::blas::matmul_$TA$TB (Dense, Dense, Dense)
#  * @brief Dense matrix multiplication: C = A^$TA B^$TB
#  * @{
#  */
# /**
#  * @brief Dense matrix multiplication: C = A^$TA B^$TB
#  * @param A Dense matrix (size M x K)
#  * @param B Dense matrix (size K x N)
#  * @param C Dense matrix (size M x N)
#  * @note
#  * - # of computation: 2MNK
#  * - Multi-threading: true
#  * - GPU acceleration: true
#  *    - # of data transfer: 0
# */ "
# for prec in double float; do
#   echo "void matmul_$TA$TB(const matrix::Dense<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
# done
# echo "/**@}*/"
# done
# done

#matmul CRS
echo "
/**
 * \defgroup mm_crs_dense monolish::blas::matmul (CRS, Dense, Dense)
 * @brief CRS and Dense matrix multiplication: C = AB
 * @{
 */
/**
 * @brief CRS and Dense matrix multiplication: C = AB
 * @param A CRS matrix (size M x K)
 * @param B Dense matrix (size K x N)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: 2*N*nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void matmul(const matrix::CRS<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

echo "
/**
 * \defgroup mm_crs_dense monolish::blas::matmul (Float, CRS, Dense, Float, Dense)
 * @brief CRS and Dense matrix multiplication: C = aAB+bC
 * @{
 */
/**
 * @brief CRS and Dense matrix multiplication: C = aAB+bC
 * @param a Float
 * @param A CRS matrix (size M x K)
 * @param B Dense matrix (size K x N)
 * @param b Float
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: 2*N*nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
*/ "
for prec in double float; do
  echo "void matmul(const $prec &a, const matrix::CRS<$prec> &A, const matrix::Dense<$prec> &B, const $prec &b, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

#matmul_* CRS
# for TA in N T; do
# for TB in N T; do
# echo "
# /**
#  * \defgroup mm_crs_dense_$TA$TB monolish::blas::matmul_$TA$TB (CRS, Dense, Dense)
#  * @brief CRS and Dense matrix multiplication: C = A^$TA B^$TB
#  * @{
#  */
# /**
#  * @brief CRS and Dense matrix multiplication: C = A^$TA B^$TB
#  * @param A CRS matrix (size M x K)
#  * @param B Dense matrix (size K x N)
#  * @param C Dense matrix (size M x N)
#  * @note
#  * - # of computation: 2*N*nnz
#  * - Multi-threading: true
#  * - GPU acceleration: true
#  *    - # of data transfer: 0
# */ "
# for prec in double float; do
#   echo "void matmul$TA$TB(const matrix::CRS<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
# done
# echo "/**@}*/"
# done
# done

#matmul LinearOperator
echo "
/**
 * \defgroup mm_LO monolish::blas::matmul (LO, LO, LO)
 * @brief LinearOperator multiplication: C = AB
 * @{
 */
/**
 * @brief LinearOperator multiplication: C = AB
 * @param A LinearOperator (size M x K)
 * @param B LinearOperator (size K x N)
 * @param C LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void matmul(const matrix::LinearOperator<$prec> &A, const matrix::LinearOperator<$prec> &B, matrix::LinearOperator<$prec> &C);"
done
echo "/**@}*/"

#matmul LinearOperator and Dense
echo "
/**
 * \defgroup mm_LO_dense monolish::blas::matmul (LO, Dense, Dense)
 * @brief LinearOperator and Dense multiplication: C = AB
 * @{
 */
/**
 * @brief LinearOperator and Dense multiplication: C = AB
 * @param A LinearOperator (size M x K)
 * @param B Dense matrix (size K x N)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: ?
 * - Multi-threading: false
 * - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void matmul(const matrix::LinearOperator<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"

#rmatmul LinearOperator and Dense
echo "
/**
 * \defgroup rmm_LO monolish::blas::rmatmul (LO, Dense, Dense)
 * @brief LinearOperator multiplication: C = A^H B
 * @{
 */
/**
 * @brief LinearOperator multiplication: C = A^H B
 * @param A LinearOperator (size K x M)
 * @param B Dense matrix (size K x N)
 * @param C Dense matrix (size M x N)
 * @note
 * - # of computation: ?
 * - Multi-threading: false
 * - GPU acceleration: false
*/ "
for prec in double float; do
  echo "void rmatmul(const matrix::LinearOperator<$prec> &A, const matrix::Dense<$prec> &B, matrix::Dense<$prec> &C);"
done
echo "/**@}*/"
echo "/**@}*/"

echo "}"
echo "}"