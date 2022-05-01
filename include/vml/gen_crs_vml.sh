#!/bin/bash
echo "//this code is generated by gen_crs_vml.sh
#pragma once

#include \"../common/monolish_common.hpp\"

/**
 * @brief
 * Vector and Matrix element-wise math library
 */
namespace monolish {
namespace vml {
"
echo "
/**
 * @addtogroup CRS_VML
 * @{
 */
"

## CRS matrix-matrix arithmetic
detail=(addition subtract multiplication division)
func=(add sub mul div)
for i in ${!detail[@]}; do
echo "
/**
 * \defgroup vml_crs${func[$i]} monolish::vml::${func[$i]}
 * @brief element by element ${detail[$i]} CRS matrix A and CRS matrix B.
 * @{
 */
/**
 * @brief element by element ${detail[$i]} CRS matrix A and
 * CRS matrix B.
 * @param A monolish CRS Matrix (size M x N)
 * @param B monolish CRS Matrix (size M x N)
 * @param C monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void ${func[$i]}(const matrix::CRS<$prec> &A, const matrix::CRS<$prec> &B, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"
done

echo ""
################################################################

## CRS matrix-scalar arithmetic
detail=(addition subtract multiplication division)
func=(add sub mul div)
for i in ${!detail[@]}; do
echo "
/**
 * \defgroup vml_scrs${func[$i]} monolish::vml::${func[$i]}
 * @brief element by element ${detail[$i]} scalar alpha and CRS matrix A.
 * @{
 */
/**
 * @brief element by element ${detail[$i]} scalar alpha and CRS matrix A.
 * @param A monolish CRS Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void ${func[$i]}(const matrix::CRS<$prec> &A, const $prec alpha, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"
done

echo ""
#############################################

## matrix-matrix pow
echo "
/**
 * \defgroup vml_crspow monolish::vml::pow
 * @brief power to CRS matrix elements (C[0:N] = pow(A[0:N], B[0:N]))
 * @{
 */
/**
 *@brief power to CRS matrix elements (C[0:N] = pow(A[0:N], B[0:N]))
 * @param A monolish CRS Matrix (size M x N)
 * @param B monolish CRS Matrix (size M x N)
 * @param C monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void pow(const matrix::CRS<$prec> &A, const matrix::CRS<$prec> &B, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"
 
echo "
/**
 * \defgroup vml_scrspow monolish::vml::pow
 * @brief power to CRS matrix elements by scalar value (C[0:N] = pow(A[0:N], alpha))
 * @{
 */
/**
 * @brief power to CRS matrix elements by scalar value (C[0:N] = pow(A[0:N], alpha))
 * @param A monolish CRS Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void pow(const matrix::CRS<$prec> &A, const $prec alpha, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"

echo ""
#############################################
## 2arg math
math=(sin sqrt sinh asin asinh tan tanh atan atanh ceil floor sign)
for math in ${math[@]}; do
echo "
/**
 * \defgroup vml_crs$math monolish::vml::$math
 * @brief $math to CRS matrix elements (C[0:nnz] = $math(A[0:nnz]))
 * @{
 */
/**
 * @brief $math to CRS matrix elements (C[0:nnz] = $math(A[0:nnz]))
 * @param A monolish CRS matrix (size M x N)
 * @param C monolish CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void $math(const matrix::CRS<$prec> &A, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"
done

echo ""
#############################################

## matrix-matrix max min
detail=(greatest smallest)
func=(max min)
for i in ${!detail[@]}; do
echo "
/**
 * \defgroup vml_crscrs${func[$i]} monolish::vml::${func[$i]}
 * @brief Create a new CRS matrix with ${detail[$i]} elements of two matrices (C[0:nnz] = ${func[$i]}(A[0:nnz], B[0:nnz]))
 * @{
 */
/**
 * @brief Create a new CRS matrix with ${detail[$i]} elements of two matrices (C[0:nnz] = ${func[$i]}(A[0:nnz], B[0:nnz]))
 * @param A monolish CRS matrix (size M x N)
 * @param B monolish CRS matrix (size M x N)
 * @param C monolish CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void ${func[$i]}(const matrix::CRS<$prec> &A, const matrix::CRS<$prec> &B, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"
done

echo ""

## CRS matrix max min
detail=(greatest smallest)
func=(max min)
for i in ${!detail[@]}; do
echo "
/**
 * \defgroup vml_crs${func[$i]} monolish::vml::${func[$i]}
 * @brief Finds the ${detail[$i]} element in CRS matrix (${func[$i]}(C[0:nnz]))
 * @{
 */
/**
 * @brief Finds the ${detail[$i]} element in CRS matrix (${func[$i]}(C[0:nnz]))
 * @param C monolish CRS matrix (size M x N)
 * @return ${detail[$i]} value
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
    echo "[[nodiscard]] $prec ${func[$i]}(const matrix::CRS<$prec> &C);"
done
echo "/**@}*/"
done

echo ""
#############################################

## reciprocal
echo "
/**
 * \defgroup vml_crsreciprocal monolish::vml::reciprocal
 * @brief reciprocal to CRS matrix elements (C[0:nnz] = 1 / A[0:nnz])
 * @{
 */
/**
 * @brief reciprocal to CRS matrix elements (C[0:nnz] = 1 / A[0:nnz])
 * @param A monolish CRS matrix (size M x N)
 * @param C monolish CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
*/ "
for prec in double float; do
  echo "void reciprocal(const matrix::CRS<$prec> &A, matrix::CRS<$prec> &C);"
done
echo "/**@}*/"

echo "}"
echo "}"
