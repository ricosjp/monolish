// this code is generated by gen_tensor_crs_vml.sh
#pragma once

#include "../common/monolish_common.hpp"

namespace monolish {
/**
 * @brief
 * Vector and Matrix element-wise math library
 */
namespace vml {

/**
 * @addtogroup tensor_CRS_VML
 * @{
 */

/**
 * \defgroup vml_crsadd monolish::vml::add
 * @brief element by element addition tensor_CRS matrix A and tensor_CRS matrix
 * B.
 * @{
 */
/**
 * @brief element by element addition tensor_CRS matrix A and
 * tensor_CRS matrix B.
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param B monolish tensor_CRS Matrix (size M x N)
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void add(const tensor::tensor_CRS<double> &A,
         const tensor::tensor_CRS<double> &B, tensor::tensor_CRS<double> &C);
void add(const tensor::tensor_CRS<float> &A, const tensor::tensor_CRS<float> &B,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crssub monolish::vml::sub
 * @brief element by element subtract tensor_CRS matrix A and tensor_CRS matrix
 * B.
 * @{
 */
/**
 * @brief element by element subtract tensor_CRS matrix A and
 * tensor_CRS matrix B.
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param B monolish tensor_CRS Matrix (size M x N)
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void sub(const tensor::tensor_CRS<double> &A,
         const tensor::tensor_CRS<double> &B, tensor::tensor_CRS<double> &C);
void sub(const tensor::tensor_CRS<float> &A, const tensor::tensor_CRS<float> &B,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsmul monolish::vml::mul
 * @brief element by element multiplication tensor_CRS matrix A and tensor_CRS
 * matrix B.
 * @{
 */
/**
 * @brief element by element multiplication tensor_CRS matrix A and
 * tensor_CRS matrix B.
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param B monolish tensor_CRS Matrix (size M x N)
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void mul(const tensor::tensor_CRS<double> &A,
         const tensor::tensor_CRS<double> &B, tensor::tensor_CRS<double> &C);
void mul(const tensor::tensor_CRS<float> &A, const tensor::tensor_CRS<float> &B,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsdiv monolish::vml::div
 * @brief element by element division tensor_CRS matrix A and tensor_CRS matrix
 * B.
 * @{
 */
/**
 * @brief element by element division tensor_CRS matrix A and
 * tensor_CRS matrix B.
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param B monolish tensor_CRS Matrix (size M x N)
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void div(const tensor::tensor_CRS<double> &A,
         const tensor::tensor_CRS<double> &B, tensor::tensor_CRS<double> &C);
void div(const tensor::tensor_CRS<float> &A, const tensor::tensor_CRS<float> &B,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_scrsadd monolish::vml::add
 * @brief element by element addition scalar alpha and tensor_CRS matrix A.
 * @{
 */
/**
 * @brief element by element addition scalar alpha and tensor_CRS matrix A.
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void add(const tensor::tensor_CRS<double> &A, const double alpha,
         tensor::tensor_CRS<double> &C);
void add(const tensor::tensor_CRS<float> &A, const float alpha,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_scrssub monolish::vml::sub
 * @brief element by element subtract scalar alpha and tensor_CRS matrix A.
 * @{
 */
/**
 * @brief element by element subtract scalar alpha and tensor_CRS matrix A.
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void sub(const tensor::tensor_CRS<double> &A, const double alpha,
         tensor::tensor_CRS<double> &C);
void sub(const tensor::tensor_CRS<float> &A, const float alpha,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_scrsmul monolish::vml::mul
 * @brief element by element multiplication scalar alpha and tensor_CRS matrix
 * A.
 * @{
 */
/**
 * @brief element by element multiplication scalar alpha and tensor_CRS matrix
 * A.
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void mul(const tensor::tensor_CRS<double> &A, const double alpha,
         tensor::tensor_CRS<double> &C);
void mul(const tensor::tensor_CRS<float> &A, const float alpha,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_scrsdiv monolish::vml::div
 * @brief element by element division scalar alpha and tensor_CRS matrix A.
 * @{
 */
/**
 * @brief element by element division scalar alpha and tensor_CRS matrix A.
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void div(const tensor::tensor_CRS<double> &A, const double alpha,
         tensor::tensor_CRS<double> &C);
void div(const tensor::tensor_CRS<float> &A, const float alpha,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crspow monolish::vml::pow
 * @brief power to tensor_CRS matrix elements (C[0:N] = pow(A[0:N], B[0:N]))
 * @{
 */
/**
 *@brief power to tensor_CRS matrix elements (C[0:N] = pow(A[0:N], B[0:N]))
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param B monolish tensor_CRS Matrix (size M x N)
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void pow(const tensor::tensor_CRS<double> &A,
         const tensor::tensor_CRS<double> &B, tensor::tensor_CRS<double> &C);
void pow(const tensor::tensor_CRS<float> &A, const tensor::tensor_CRS<float> &B,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_scrspow monolish::vml::pow
 * @brief power to tensor_CRS matrix elements by scalar value (C[0:N] =
 * pow(A[0:N], alpha))
 * @{
 */
/**
 * @brief power to tensor_CRS matrix elements by scalar value (C[0:N] =
 * pow(A[0:N], alpha))
 * @param A monolish tensor_CRS Matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish tensor_CRS Matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void pow(const tensor::tensor_CRS<double> &A, const double alpha,
         tensor::tensor_CRS<double> &C);
void pow(const tensor::tensor_CRS<float> &A, const float alpha,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crssin monolish::vml::sin
 * @brief sin to tensor_CRS matrix elements (C[0:nnz] = sin(A[0:nnz]))
 * @{
 */
/**
 * @brief sin to tensor_CRS matrix elements (C[0:nnz] = sin(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void sin(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void sin(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crssqrt monolish::vml::sqrt
 * @brief sqrt to tensor_CRS matrix elements (C[0:nnz] = sqrt(A[0:nnz]))
 * @{
 */
/**
 * @brief sqrt to tensor_CRS matrix elements (C[0:nnz] = sqrt(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void sqrt(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void sqrt(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crssinh monolish::vml::sinh
 * @brief sinh to tensor_CRS matrix elements (C[0:nnz] = sinh(A[0:nnz]))
 * @{
 */
/**
 * @brief sinh to tensor_CRS matrix elements (C[0:nnz] = sinh(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void sinh(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void sinh(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsasin monolish::vml::asin
 * @brief asin to tensor_CRS matrix elements (C[0:nnz] = asin(A[0:nnz]))
 * @{
 */
/**
 * @brief asin to tensor_CRS matrix elements (C[0:nnz] = asin(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void asin(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void asin(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsasinh monolish::vml::asinh
 * @brief asinh to tensor_CRS matrix elements (C[0:nnz] = asinh(A[0:nnz]))
 * @{
 */
/**
 * @brief asinh to tensor_CRS matrix elements (C[0:nnz] = asinh(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void asinh(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void asinh(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crstan monolish::vml::tan
 * @brief tan to tensor_CRS matrix elements (C[0:nnz] = tan(A[0:nnz]))
 * @{
 */
/**
 * @brief tan to tensor_CRS matrix elements (C[0:nnz] = tan(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void tan(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void tan(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crstanh monolish::vml::tanh
 * @brief tanh to tensor_CRS matrix elements (C[0:nnz] = tanh(A[0:nnz]))
 * @{
 */
/**
 * @brief tanh to tensor_CRS matrix elements (C[0:nnz] = tanh(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void tanh(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void tanh(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsatan monolish::vml::atan
 * @brief atan to tensor_CRS matrix elements (C[0:nnz] = atan(A[0:nnz]))
 * @{
 */
/**
 * @brief atan to tensor_CRS matrix elements (C[0:nnz] = atan(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void atan(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void atan(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsatanh monolish::vml::atanh
 * @brief atanh to tensor_CRS matrix elements (C[0:nnz] = atanh(A[0:nnz]))
 * @{
 */
/**
 * @brief atanh to tensor_CRS matrix elements (C[0:nnz] = atanh(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void atanh(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void atanh(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsceil monolish::vml::ceil
 * @brief ceil to tensor_CRS matrix elements (C[0:nnz] = ceil(A[0:nnz]))
 * @{
 */
/**
 * @brief ceil to tensor_CRS matrix elements (C[0:nnz] = ceil(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void ceil(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void ceil(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsfloor monolish::vml::floor
 * @brief floor to tensor_CRS matrix elements (C[0:nnz] = floor(A[0:nnz]))
 * @{
 */
/**
 * @brief floor to tensor_CRS matrix elements (C[0:nnz] = floor(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void floor(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void floor(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crssign monolish::vml::sign
 * @brief sign to tensor_CRS matrix elements (C[0:nnz] = sign(A[0:nnz]))
 * @{
 */
/**
 * @brief sign to tensor_CRS matrix elements (C[0:nnz] = sign(A[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void sign(const tensor::tensor_CRS<double> &A, tensor::tensor_CRS<double> &C);
void sign(const tensor::tensor_CRS<float> &A, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crscrsmax monolish::vml::max
 * @brief Create a new tensor_CRS matrix with greatest elements of two matrices
 * (C[0:nnz] = max(A[0:nnz], B[0:nnz]))
 * @{
 */
/**
 * @brief Create a new tensor_CRS matrix with greatest elements of two matrices
 * (C[0:nnz] = max(A[0:nnz], B[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param B monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void max(const tensor::tensor_CRS<double> &A,
         const tensor::tensor_CRS<double> &B, tensor::tensor_CRS<double> &C);
void max(const tensor::tensor_CRS<float> &A, const tensor::tensor_CRS<float> &B,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crscrsmin monolish::vml::min
 * @brief Create a new tensor_CRS matrix with smallest elements of two matrices
 * (C[0:nnz] = min(A[0:nnz], B[0:nnz]))
 * @{
 */
/**
 * @brief Create a new tensor_CRS matrix with smallest elements of two matrices
 * (C[0:nnz] = min(A[0:nnz], B[0:nnz]))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param B monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void min(const tensor::tensor_CRS<double> &A,
         const tensor::tensor_CRS<double> &B, tensor::tensor_CRS<double> &C);
void min(const tensor::tensor_CRS<float> &A, const tensor::tensor_CRS<float> &B,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_scrsmax monolish::vml::max
 * @brief Create a new tensor_CRS matrix with greatest elements of tensor_CRS
 * matrix or scalar (C[0:nnz] = max(A[0:nnz], alpha))
 * @{
 */
/**
 * @brief Create a new tensor_CRS matrix with greatest elements of tensor_CRS
 * matrix or scalar (C[0:nnz] = max(A[0:nnz], alpha))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */
void max(const tensor::tensor_CRS<double> &A, const double alpha,
         tensor::tensor_CRS<double> &C);
void max(const tensor::tensor_CRS<float> &A, const float alpha,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_scrsmin monolish::vml::min
 * @brief Create a new tensor_CRS matrix with smallest elements of tensor_CRS
 * matrix or scalar (C[0:nnz] = min(A[0:nnz], alpha))
 * @{
 */
/**
 * @brief Create a new tensor_CRS matrix with smallest elements of tensor_CRS
 * matrix or scalar (C[0:nnz] = min(A[0:nnz], alpha))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param alpha scalar value
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */
void min(const tensor::tensor_CRS<double> &A, const double alpha,
         tensor::tensor_CRS<double> &C);
void min(const tensor::tensor_CRS<float> &A, const float alpha,
         tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsmax monolish::vml::max
 * @brief Finds the greatest element in tensor_CRS matrix (max(C[0:nnz]))
 * @{
 */
/**
 * @brief Finds the greatest element in tensor_CRS matrix (max(C[0:nnz]))
 * @param C monolish tensor_CRS matrix (size M x N)
 * @return greatest value
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
[[nodiscard]] double max(const tensor::tensor_CRS<double> &C);
[[nodiscard]] float max(const tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsmin monolish::vml::min
 * @brief Finds the smallest element in tensor_CRS matrix (min(C[0:nnz]))
 * @{
 */
/**
 * @brief Finds the smallest element in tensor_CRS matrix (min(C[0:nnz]))
 * @param C monolish tensor_CRS matrix (size M x N)
 * @return smallest value
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
[[nodiscard]] double min(const tensor::tensor_CRS<double> &C);
[[nodiscard]] float min(const tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_scrsalo monolish::vml::alo
 * @brief Asymmetric linear operation to tensor_CRS matrix elements (C[0:nnz] =
 * alpha max(A[0:nnz], 0) + beta min(A[0:nnz], 0))
 * @{
 */
/**
 * @brief Asymmetric linear operation to tensor_CRS matrix elements (C[0:nnz] =
 * alpha max(A[0:nnz], 0) + beta min(A[0:nnz], 0))
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param alpha linear coefficient in positive range
 * @param beta linear coefficient in negative range
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: M*N
 * - Multi-threading: true
 * - GPU acceleration: true
 */
void alo(const tensor::tensor_CRS<double> &A, const double alpha,
         const double beta, tensor::tensor_CRS<double> &C);
void alo(const tensor::tensor_CRS<float> &A, const float alpha,
         const float beta, tensor::tensor_CRS<float> &C);
/**@}*/

/**
 * \defgroup vml_crsreciprocal monolish::vml::reciprocal
 * @brief reciprocal to tensor_CRS matrix elements (C[0:nnz] = 1 / A[0:nnz])
 * @{
 */
/**
 * @brief reciprocal to tensor_CRS matrix elements (C[0:nnz] = 1 / A[0:nnz])
 * @param A monolish tensor_CRS matrix (size M x N)
 * @param C monolish tensor_CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B, and C must be same non-zero structure
 */
void reciprocal(const tensor::tensor_CRS<double> &A,
                tensor::tensor_CRS<double> &C);
void reciprocal(const tensor::tensor_CRS<float> &A,
                tensor::tensor_CRS<float> &C);
/**@}*/
} // namespace vml
} // namespace monolish
