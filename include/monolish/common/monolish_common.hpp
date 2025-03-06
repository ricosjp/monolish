#pragma once
#if USE_SXAT
#undef _HAS_CPP17
#endif
#include "monolish_dense.hpp"
#include "monolish_logger.hpp"
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"
#include "monolish_view1D.hpp"
#include "monolish_view_dense.hpp"
#include "monolish_view_tensor_dense.hpp"
#include <initializer_list>

// error code
#define MONOLISH_SOLVER_SUCCESS 0
#define MONOLISH_SOLVER_SIZE_ERROR -1
#define MONOLISH_SOLVER_MAXITER -2
#define MONOLISH_SOLVER_BREAKDOWN -3
#define MONOLISH_SOLVER_RESIDUAL_NAN -4
#define MONOLISH_SOLVER_NOT_IMPL -10

/**
 * @brief monolish namespaces
 */
namespace monolish {
/**
 * @brief monolish utilities
 */
namespace util {
/**
 * @addtogroup utils
 * @{
 */

/**
 * @brief get the number of devices
 * @return the number of devices (If the device is not found or the GPU is not
 * enabled, return value is negative)
 * @ingroup GPUutil
 */
int get_num_devices();

/**
 * @brief set default device number
 * @return if the GPU is not enabled, return false
 * @ingroup GPUutil
 */
bool set_default_device(size_t device_num);

/**
 * @brief get default device number
 * @return the device number (If the device is not found or the GPU is not
 * enabled, return value is negative)
 * @ingroup GPUutil
 */
int get_default_device();

/**
 * @addtogroup Other
 * @{
 */

/**
 * @brief get nrm |b-Ax|_2
 * @param A Dense matrix (size M x N)
 * @param x monolish vector (size N)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: 2*M*nnz + N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<vector<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);

float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
                      const vector<float> &y);
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
                      const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
                      const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
                      const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<vector<float>, float> &x,
                      const vector<float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<vector<float>, float> &x,
                      const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<vector<float>, float> &x,
                      const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<vector<float>, float> &x,
                      const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<matrix::Dense<float>, float> &x,
                      const vector<float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<matrix::Dense<float>, float> &x,
                      const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<matrix::Dense<float>, float> &x,
                      const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<matrix::Dense<float>, float> &x,
                      const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<tensor::tensor_Dense<float>, float> &x,
                      const vector<float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<tensor::tensor_Dense<float>, float> &x,
                      const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<tensor::tensor_Dense<float>, float> &x,
                      const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
                      const view1D<tensor::tensor_Dense<float>, float> &x,
                      const view1D<tensor::tensor_Dense<float>, float> &y);

/**
 * @brief get nrm |b-Ax|_2
 * @param A CRS matrix (size M x N)
 * @param x monolish vector (size N)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: 2*M*nnz + N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<vector<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);

float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
                      const vector<float> &y);
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
                      const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
                      const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
                      const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<vector<float>, float> &x,
                      const vector<float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<vector<float>, float> &x,
                      const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<vector<float>, float> &x,
                      const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<vector<float>, float> &x,
                      const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<matrix::Dense<float>, float> &x,
                      const vector<float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<matrix::Dense<float>, float> &x,
                      const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<matrix::Dense<float>, float> &x,
                      const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<matrix::Dense<float>, float> &x,
                      const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<tensor::tensor_Dense<float>, float> &x,
                      const vector<float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<tensor::tensor_Dense<float>, float> &x,
                      const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<tensor::tensor_Dense<float>, float> &x,
                      const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
                      const view1D<tensor::tensor_Dense<float>, float> &x,
                      const view1D<tensor::tensor_Dense<float>, float> &y);

/**
 * @brief get nrm |b-Ax|_2
 * @param A LinearOperator (size M x N)
 * @param x monolish vector (size N)
 * @param y monolish vector (size M)
 * @note
 * - # of computation: 2*M*nnz + N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const vector<double> &x, const vector<double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const vector<double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const vector<double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const vector<double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<vector<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<vector<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<matrix::Dense<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const vector<double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
                       const view1D<tensor::tensor_Dense<double>, double> &x,
                       const view1D<tensor::tensor_Dense<double>, double> &y);

/**@}*/

/**
 * @brief check error
 * @param[in] err solver err code
 * @ingroup errcheck
 * @note
 * - MONOLISH_SOLVER_SUCCESS 0
 * - MONOLISH_SOLVER_SIZE_ERROR -1
 * - MONOLISH_SOLVER_MAXITER -2
 * - MONOLISH_SOLVER_BREAKDOWN -3
 * - MONOLISH_SOLVER_RESIDUAL_NAN -4
 * - MONOLISH_SOLVER_NOT_IMPL -10
 */
[[nodiscard]] bool solver_check(const int err);

// logger

/**
 * @brief Specifying the log level
 * @param Level loglevel
 * @note loglevel is
 * 1. logging solvers (CG, Jacobi, LU...etc.)
 * 2. logging solvers and BLAS functions (matmul, matvec, arithmetic
 *operators..etc.)
 * 3. logging solvers and BLAS functions and utils (send, recv,
 *allocation...etc.)
 * @ingroup logger
 **/
void set_log_level(const size_t Level);

/**
 * @brief Specifying the log finename
 * @param filename the log filename
 * @ingroup logger
 **/
void set_log_filename(const std::string filename);

// create typical data///////////////////////////

/**
 * @brief create random vector
 * @param vec allocated vector
 * @param min min. of random
 * @param max min. of random
 * @note the ramdom number generator is random generator is mt19937
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
void random_vector(vector<T> &vec, const T min, const T max);

/**
 * @brief create random vector with random seed
 * @param vec allocated vector
 * @param min min. of random
 * @param max min. of random
 * @param seed random seed
 * @note the ramdom number generator is random generator is mt19937
 * @note
 * - # of computation: N
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
void random_vector(vector<T> &vec, const T min, const T max,
                   const std::uint32_t seed);

// is_same //////////////////

/**
 * @brief compare matrix structure
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_structure(const T A, const U B) {
  return false;
}

/**
 * @brief compare structure of vector (same as is_same_size())
 * @param x monolish vector
 * @param y monolish vector
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_structure(const vector<T> &x, const vector<T> &y) {
  return x.size() == y.size();
}

/**
 * @brief compare structure of vector (same as is_same_size())
 * @param x monolish vector
 * @param y monolish vector
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_structure(const view1D<T, U> &x,
                                     const vector<U> &y) {
  return x.size() == y.size();
}

/**
 * @brief compare structure of vector (same as is_same_size())
 * @param x monolish vector
 * @param y monolish vector
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_structure(const vector<U> &x,
                                     const view1D<T, U> &y) {
  return x.size() == y.size();
}

/**
 * @brief compare structure of vector (same as is_same_size())
 * @param x monolish vector
 * @param y monolish vector
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U, typename V>
[[nodiscard]] bool is_same_structure(const view1D<T, V> &x,
                                     const view1D<U, V> &y) {
  return x.size() == y.size();
}

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_structure(const matrix::Dense<T> &A,
                                     const matrix::Dense<T> &B);

/**
 * @brief compare structure using col_index and row_index, M, and N
 * @param A COO matrix
 * @param B COO matrix
 * @return true is same structure
 * @note
 * - # of computation: 2nnz
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_structure(const matrix::COO<T> &A,
                                     const matrix::COO<T> &B);

/**
 * @brief compare structure using structure_hash, M, and N
 * @param A CRS matrix
 * @param B CRS matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_structure(const matrix::CRS<T> &A,
                                     const matrix::CRS<T> &B);

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A LinearOperator matrix
 * @param B LinearOperator matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_structure(const matrix::LinearOperator<T> &A,
                                     const matrix::LinearOperator<T> &B);

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_structure(const view_Dense<T, U> &A,
                                     const matrix::Dense<U> &B);

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_structure(const matrix::Dense<U> &A,
                                     const view_Dense<T, U> &B);

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U, typename V>
[[nodiscard]] bool is_same_structure(const view_Dense<T, V> &A,
                                     const view_Dense<U, V> &B);

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_structure(const tensor::tensor_Dense<T> &A,
                                     const tensor::tensor_Dense<T> &B);

/**
 * @brief compare structure using col_index and row_index, M, and N
 * @param A COO matrix
 * @param B COO matrix
 * @return true is same structure
 * @note
 * - # of computation: 2nnz
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_structure(const tensor::tensor_COO<T> &A,
                                     const tensor::tensor_COO<T> &B);

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_structure(const view_tensor_Dense<T, U> &A,
                                     const tensor::tensor_Dense<U> &B);

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_structure(const tensor::tensor_Dense<U> &A,
                                     const view_tensor_Dense<T, U> &B);

/**
 * @brief compare structure using M and N (same as is_same_size())
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same structure
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U, typename V>
[[nodiscard]] bool is_same_structure(const view_tensor_Dense<T, V> &A,
                                     const view_tensor_Dense<U, V> &B);

/**
 * @brief compare matrix structure
 * @ingroup compare
 **/
template <typename T, typename U, typename V, typename... types>
[[nodiscard]] bool is_same_structure(const T &A, const U &B, const V &C,
                                     const types &...args) {
  return is_same_structure(A, B) && is_same_structure(A, C, args...);
}

/**
 * @brief compare size of vector or 1Dview (same as is_same_structure())
 * @param x monolish vector
 * @param y monolish vector
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_size(const T &x, const U &y) {
  return x.size() == y.size();
}

/**
 * @brief compare row and col size
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_size(const matrix::Dense<T> &A,
                                const matrix::Dense<T> &B);

/**
 * @brief compare row and col size
 * @param A COO matrix
 * @param B COO matrix
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_size(const matrix::COO<T> &A,
                                const matrix::COO<T> &B);

/**
 * @brief compare row and col size
 * @param A COO matrix
 * @param B COO matrix
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_size(const matrix::CRS<T> &A,
                                const matrix::CRS<T> &B);

/**
 * @brief compare row and col size
 * @param A LinearOperator matrix
 * @param B LinearOperator matrix
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_size(const matrix::LinearOperator<T> &A,
                                const matrix::LinearOperator<T> &B);

/**
 * @brief compare row and col size
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_size(const view_Dense<T, U> &A,
                                const matrix::Dense<U> &B);
/**
 * @brief compare row and col size
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_size(const matrix::Dense<U> &A,
                                const view_Dense<T, U> &B);
/**
 * @brief compare row and col size
 * @param A Dense matrix
 * @param B Dense matrix
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U, typename V>
[[nodiscard]] bool is_same_size(const view_Dense<T, V> &A,
                                const view_Dense<U, V> &B);

/**
 * @brief compare shape
 * @param A Dense tensor
 * @param B Dense tensor
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_size(const tensor::tensor_Dense<T> &A,
                                const tensor::tensor_Dense<T> &B);

/**
 * @brief compare shape
 * @param A COO tensor
 * @param B COO tensor
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_size(const tensor::tensor_COO<T> &A,
                                const tensor::tensor_COO<T> &B);

/**
 * @brief compare shape
 * @param A COO tensor
 * @param B COO tensor
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T>
[[nodiscard]] bool is_same_size(const tensor::tensor_CRS<T> &A,
                                const tensor::tensor_CRS<T> &B);

/**
 * @brief compare shape
 * @param A Dense tensor
 * @param B Dense tensor
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_size(const view_tensor_Dense<T, U> &A,
                                const tensor::tensor_Dense<U> &B);
/**
 * @brief compare shape
 * @param A Dense tensor
 * @param B Dense tensor
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_size(const tensor::tensor_Dense<U> &A,
                                const view_tensor_Dense<T, U> &B);
/**
 * @brief compare shape
 * @param A Dense tensor
 * @param B Dense tensor
 * @return true is same size
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U, typename V>
[[nodiscard]] bool is_same_size(const view_tensor_Dense<T, V> &A,
                                const view_tensor_Dense<U, V> &B);

/**
 * @brief compare matrix size
 * @ingroup compare
 **/
template <typename T, typename U, typename V, typename... types>
[[nodiscard]] bool is_same_size(const T &arg1, const U &arg2, const V &arg3,
                                const types &...args) {
  return is_same_size(arg1, arg2) && is_same_size(arg1, arg3, args...);
}

/**
 * @brief compare same device memory status
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U>
[[nodiscard]] bool is_same_device_mem_stat(const T &arg1, const U &arg2) {
  return arg1.get_device_mem_stat() == arg2.get_device_mem_stat();
}

/**
 * @brief compare same device memory status
 * @note
 * - # of computation: 1
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup compare
 **/
template <typename T, typename U, typename... types>
[[nodiscard]] bool is_same_device_mem_stat(const T &arg1, const U &arg2,
                                           const types &...args) {
  return is_same_device_mem_stat(arg1, arg2) &&
         is_same_device_mem_stat(arg1, args...);
}

// create matrix //////////////////

/**
 * @brief create band matrix
 * @param M # of Row
 * @param N # of col.
 * @param W half-bandwidth (bandwidth is 2*W+1)
 * @param diag_val value of diagonal elements
 * @param val value of non-diagonal elements
 * @note
 * - # of computation: M*W
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] matrix::COO<T> band_matrix(const int M, const int N, const int W,
                                         const T diag_val, const T val);

/**
 * @brief create asymmetric band matrix
 * @param M # of Row
 * @param N # of col.
 * @param W half-bandwidth (bandwidth is 2*W+1)
 * @param diag_val value of diagonal elements
 * @param Uval value of upper non-diagonal elements
 * @param Lval value of lower non-diagonal elements
 * @note
 * - # of computation: M*W
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] matrix::COO<T> asym_band_matrix(const int M, const int N,
                                              const int W, const T diag_val,
                                              const T Uval, const T Lval);

/**
 * @brief create random structure matrix (column number is decided by random)
 * @param M # of Row
 * @param N # of col.
 * @param nnzrow non-zero elements per row
 * @param val value of elements
 * @note
 * - # of computation: M*nnzrow
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] matrix::COO<T> random_structure_matrix(const int M, const int N,
                                                     const int nnzrow,
                                                     const T val);

/**
 * @brief create random structure tensor (index is decided by random)
 * @param M # of shape 1
 * @param N # of shape 2
 * @param nnzrow non-zero elements per row
 * @param val value of elements
 * @note
 * - # of computation: M*N*nnzrow
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] tensor::tensor_COO<T>
random_structure_tensor(const size_t M, const size_t N, const size_t nnzrow,
                        const T val);

/**
 * @brief create random structure tensor (index is decided by random)
 * @param M # of shape 1
 * @param N # of shape 2
 * @param L # of shape 3
 * @param nnzrow non-zero elements per row
 * @param val value of elements
 * @note
 * - # of computation: M*N*nnzrow
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] tensor::tensor_COO<T>
random_structure_tensor(const size_t M, const size_t N, const size_t L,
                        const size_t nnzrow, const T val);

/**
 * @brief create random structure tensor (index is decided by random)
 * @param M # of shape 1
 * @param N # of shape 2
 * @param L # of shape 3
 * @param nnzrow non-zero elements per row
 * @param val value of elements
 * @note
 * - # of computation: M*N*nnzrow
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] tensor::tensor_COO<T>
random_structure_tensor(const size_t K, const size_t M, const size_t N,
                        const size_t L, const size_t nnzrow, const T val);

/**
 * @brief create band matrix
 * @param M # of Row and col.
 * @note
 * - # of computation: M
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T> [[nodiscard]] matrix::COO<T> eye(const int M);

/**
 * @brief create Frank matrix
 * @param M # of row and col
 * @note
 * - # of computation: M^2
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T> [[nodiscard]] matrix::COO<T> frank_matrix(const int &M);

/**
 * @brief Nth eigenvalue from the bottom of MxM Frank matrix
 * @param M dimension of Frank matrix
 * @param N #-th eigenvalue from the bottom
 * @note
 * - # of computation: O(1)
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] T frank_matrix_eigenvalue(const int &M, const int &N);

/**
 * @brief create tridiagonal Toeplitz matrix
 * @param M # of row and col
 * @param a value of diagonal elements
 * @param b value of next-to-diagonal elements
 * @note
 * - # of computation: M
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] matrix::COO<T> tridiagonal_toeplitz_matrix(const int &M, T a,
                                                         T b);

/**
 * @brief Nth smallest eigenvalue of MxM tridiagonal Toeplitz matrix
 * @param M dimension of tridiagonal Toeplitz matrix
 * @param N #-th eigenvalue from the bottom
 * @param a value of diagonal elements
 * @param b value of next-to-diagonal elements
 * @note
 * - # of computation: O(1)
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] T tridiagonal_toeplitz_matrix_eigenvalue(const int &M, int N, T a,
                                                       T b);

/**
 * @brief create 1D Laplacian matrix
 * @param M # of row and col
 * @note
 * - # of computation: M
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] matrix::COO<T> laplacian_matrix_1D(const int &M);

/**
 * @brief Nth smallest eigenvalue of 1D Laplacian matrix
 * @param M dimension of tridiagonal Toeplitz matrix
 * @param N #-th eigenvalue from the bottom
 * @note
 * - # of computation: O(1)
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] T laplacian_matrix_1D_eigenvalue(const int &M, int N);

/**
 * @brief create two dimensional Laplacian matrix using the five point central
 *difference scheme
 * @param M # of grid point
 * @param N # of grid point
 * @note
 * - # of computation: N*M
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 **/
template <typename T>
[[nodiscard]] matrix::COO<T> laplacian_matrix_2D_5p(const int M, const int N);

/**
 * @brief create Toeplitz-plus-Hankel matrix
 * @param M # of row and col
 * @param a0 value of diagonal elements
 * @param a1 value of next-to-diagonal elements
 * @param a2 value of second-next-to-diagonal elements
 * @note
 * - taken from arxiv:2007.08130
 * - (0, 0) and (M-1, M-1) elements are modified to a0-a2
 * - both A and B of the GEVP have same structure
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 */
template <typename T>
[[nodiscard]] matrix::COO<T> toeplitz_plus_hankel_matrix(const int &M, T a0,
                                                         T a1, T a2);

/**
 * @brief Nth smallest eigenvalue of GEVP Ax=lBx of Toeplitz-plus-Hankel
 * matrixes A, B
 * @param M dimension of Toeplitz-plus-Hankel marices
 * @param N #-th eigenvalue from the bottom
 * @param a0, a1, a2 value of Toeplitz-plus-Hankel matrix A
 * @param b0, b1, b2 value of TOeplitz-plus-Hankel matrix B
 * @note
 * - # of computation: O(1)
 * - Multi-threading: false
 * - GPU acceleration: false
 * @ingroup gendata
 */
template <typename T>
[[nodiscard]] T toeplitz_plus_hankel_matrix_eigenvalue(const int &M, int N,
                                                       T a0, T a1, T a2, T b0,
                                                       T b1, T b2);

// send///////////////////

/**
 * @brief send data to GPU
 * @ingroup GPUutil
 **/
template <typename T> void send(T &x) { x.send(); }

/**
 * @brief send data to GPU
 * @ingroup GPUutil
 **/
template <typename T, typename... Types> void send(T &x, Types &...args) {
  x.send();
  send(args...);
}

// recv///////////////////
/**
 * @brief recv. and free data from GPU
 * @ingroup GPUutil
 **/
template <typename T> void recv(T &x) { x.recv(); }

/**
 * @brief recv. and free data to GPU
 * @ingroup GPUutil
 **/
template <typename T, typename... Types> void recv(T &x, Types &...args) {
  x.recv();
  recv(args...);
}

// device_free///////////////////

/**
 * @brief free data of GPU
 * @ingroup GPUutil
 **/
template <typename T> void device_free(T &x) { x.device_free(); }

/**
 * @brief free data of GPU
 * @ingroup GPUutil
 **/
template <typename T, typename... Types>
void device_free(T &x, Types &...args) {
  x.device_free();
  device_free(args...);
}

/**
 * @brief get build option (true: with avx, false: without avx)
 * @ingroup build_options
 **/
[[nodiscard]] bool build_with_avx();

/**
 * @brief get build option (true: with avx2, false: without avx2)
 * @ingroup build_options
 **/
[[nodiscard]] bool build_with_avx2();

/**
 * @brief get build option (true: with avx512, false: without avx512)
 * @ingroup build_options
 **/
[[nodiscard]] bool build_with_avx512();

/**
 * @brief get build option (true: enable MPI, false: disable MPI)
 * @ingroup build_options
 **/
[[nodiscard]] bool build_with_mpi();

/**
 * @brief get build option (true: enable gpu, false: disable gpu)
 * @ingroup build_options
 **/
[[nodiscard]] bool build_with_gpu();

/**
 * @brief get build option (true: with intel mkl, false: without intel mkl)
 * @ingroup build_options
 **/
[[nodiscard]] bool build_with_mkl();

/**
 * @brief get build option (true: with lapack, false: without lapack (=with
 *intel mkl))
 * @ingroup build_options
 **/
[[nodiscard]] bool build_with_lapack();

/**
 * @brief get build option (true: with cblas, false: without cblas (=with intel
 *mkl))
 * @ingroup build_options
 **/
[[nodiscard]] bool build_with_cblas();
/**@}*/

} // namespace util
} // namespace monolish
