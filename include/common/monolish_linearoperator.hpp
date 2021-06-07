/**
 * @author fockl
 * @file monolish_linearoperator.h
 * @brief declare linearoperator class
 * @date 2021
 **/

#pragma once
#include <exception>
#include <functional>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

#if USE_SXAT
#undef _HAS_CPP17
#endif
#include <random>
#if USE_SXAT
#define _HAS_CPP17 1
#endif

namespace monolish {
template <typename Float> class vector;
namespace matrix {
template <typename Float> class Dense;
template <typename Float> class COO;
template <typename Float> class CRS;

/**
 * @brief Linear Operator imitating Matrix
 * @note
 * - Multi-threading: depends on matvec/rmatvec functions
 * - GPU acceleration: depends on matvec/rmatvec functions
 */
template <typename Float> class LinearOperator {
private:
  /**
   * @brief # of row
   */
  size_t rowN;

  /**
   * @brief # of col
   */
  size_t colN;

  /**
   * @brief true: sended, false: not send
   */
  mutable bool gpu_status = false;

  /**
   * @brief pseudo multiplication function of matrix and vector
   */
  std::function<vector<Float>(const vector<Float> &)> matvec = nullptr;

  /**
   * @brief pseudo multiplication function of (Hermitian) transposed matrix and
   * vector
   */
  std::function<vector<Float>(const vector<Float> &)> rmatvec = nullptr;

  /**
   * @brief pseudo multiplication function of matrix and dense matrix
   */
  std::function<Dense<Float>(const Dense<Float> &)> matmul_dense = nullptr;

  /**
   * @brief pseudo multiplication function of (Hermitian) transposed matrix and
   * dense matrix
   */
  std::function<Dense<Float>(const Dense<Float> &)> rmatmul_dense = nullptr;

public:
  LinearOperator() {}

  /**
   * @brief declare LinearOperator
   * @param M # of row
   * @param N # of col
   * @note
   * - # of computation: 4
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  LinearOperator(const size_t M, const size_t N);

  /**
   * @brief declare LinearOperator
   * @param M # of row
   * @param N # of col
   * @param MATVEC multiplication function of matrix and vector
   * @note
   * - # of computation: 4 + 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  LinearOperator(
      const size_t M, const size_t N,
      const std::function<vector<Float>(const vector<Float> &)> &MATVEC);

  /**
   * @brief declare LinearOperator
   * @param M # of row
   * @param N # of col
   * @param MATVEC multiplication function of matrix and vector
   * @param RMATVEC multiplication function of (Hermitian) transposed matrix and
   * vector
   * @note
   * - # of computation: 4 + 2 functions
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  LinearOperator(
      const size_t M, const size_t N,
      const std::function<vector<Float>(const vector<Float> &)> &MATVEC,
      const std::function<vector<Float>(const vector<Float> &)> &RMATVEC);

  /**
   * @brief declare LinearOperator
   * @param M # of row
   * @param N # of col
   * @param MATMUL multiplication function of matrix and matrix
   * @note
   * - # of computation: 4 + 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  LinearOperator(
      const size_t M, const size_t N,
      const std::function<Dense<Float>(const Dense<Float> &)> &MATMUL);

  /**
   * @brief declare LinearOperator
   * @param M # of row
   * @param N # of col
   * @param MATMUL multiplication function of matrix and matrix
   * @param RMATMUL multiplication function of (Hermitian) transposed  matrix
   * and matrix
   * @note
   * - # of computation: 4 + 2 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  LinearOperator(
      const size_t M, const size_t N,
      const std::function<Dense<Float>(const Dense<Float> &)> &MATMUL,
      const std::function<Dense<Float>(const Dense<Float> &)> &RMATMUL);

  /**
   * @brief Convert LinearOperator from COO
   * @param coo COO format matrix
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void convert(COO<Float> &coo);

  /**
   * @brief Create LinearOperator from COO
   * @param coo Source COO format matrix
   * @return coo COO format matrix
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  LinearOperator(COO<Float> &coo) { convert(coo); }

  void convert(CRS<Float> &crs);

  LinearOperator(CRS<Float> &crs) { convert(crs); }

  void convert(Dense<Float> &dense);

  LinearOperator(Dense<Float> &dense) { convert(dense); }

  void convert_to_Dense(Dense<Float> &dense) const;

  /**
   * @brief Create LinearOperator from LinearOperator
   * @param LinearOperator format LinearOperator
   * @note
   * - # of computation: 4 + 2 functions
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  LinearOperator(const LinearOperator<Float> &linearoperator);

  /**
   * @brief get # of row
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] size_t get_row() const { return rowN; }

  /**
   * @brief get # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] size_t get_col() const { return colN; }

  /**
   * @brief get multiplication function of matrix and vector
   * @note
   * - # of computation: 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] std::function<vector<Float>(const vector<Float> &)>
  get_matvec() const {
    return matvec;
  }

  /**
   * @brief get multiplication function of (Hermitian) transposed matrix and
  vector C = A;
   * @note
   * - # of computation: 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] std::function<vector<Float>(const vector<Float> &)>
  get_rmatvec() const {
    return rmatvec;
  }

  /**
   * @brief get multiplication function of matrix and matrix dense
   * @note
   * - # of computation: 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] std::function<
      matrix::Dense<Float>(const matrix::Dense<Float> &)>
  get_matmul_dense() const {
    return matmul_dense;
  }

  /**
   * @brief get multiplication function of (Hermitian) transposed matrix and
  matrix dense;
   * @note
   * - # of computation: 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] std::function<
      matrix::Dense<Float>(const matrix::Dense<Float> &)>
  get_rmatmul_dense() const {
    return rmatmul_dense;
  }

  /**
   * @brief get flag that shows matvec is defined or not
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] bool get_matvec_init_flag() const {
    return !(matvec == nullptr);
  }

  /**
   * @brief get flag that shows rmatvec is defined or not
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] bool get_rmatvec_init_flag() const {
    return !(rmatvec == nullptr);
  }

  /**
   * @brief get flag that shows matmul_dense is defined or not
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] bool get_matmul_dense_init_flag() const {
    return !(matmul_dense == nullptr);
  }

  /**
   * @brief get flag that shows rmatmul_dense is defined or not
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  [[nodiscard]] bool get_rmatmul_dense_init_flag() const {
    return !(rmatmul_dense == nullptr);
  }

  /**
   * @brief set multiplication function of matrix and vector
   * @note
   * - # of computation: 1 + 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void
  set_matvec(const std::function<vector<Float>(const vector<Float> &)> &MATVEC);

  /**
   * @brief set multiplication function of (Hermitian) transposed matrix and
   * vector
   * @note
   * - # of computation: 1 + 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void set_rmatvec(
      const std::function<vector<Float>(const vector<Float> &)> &RMATVEC);

  /**
   * @brief set multiplication function of matrix and matrix dense
   * @note
   * - # of computation: 1 + 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void set_matmul_dense(
      const std::function<matrix::Dense<Float>(const matrix::Dense<Float> &)>
          &MATMUL);

  /**
   * @brief set multiplication function of (Hermitian) transposed matrix and
   * matrix dense
   * @note
   * - # of computation: 1 + 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void set_rmatmul_dense(
      const std::function<matrix::Dense<Float>(const matrix::Dense<Float> &)>
          &RMATMUL);

  /**
   * @brief get format name "LinearOperator"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const { return "LinearOperator"; }

  /**
   * @brief send data to GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void send() const {};

  /**
   * @brief recv. data to GPU, and free data on GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void recv() const {};

  /**
   * @brief recv. data to GPU (w/o free)
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void nonfree_recv() const {};

  /**
   * @brief free data on GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void device_free() const {};

  /**
   * @brief true: sended, false: not send
   * @return gpu status
   * **/
  [[nodiscard]] bool get_device_mem_stat() const { return gpu_status; };

  void set_device_mem_stat(bool status) {
    gpu_status = status;
    return;
  };

  /**
   * @brief destructor of LinearOperator, free GPU memory
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   * **/
  ~LinearOperator() {}

  /**
   * @brief get diag. vector
   * @param vec diag. vector
   **/
  void diag(vector<Float> &vec) const;
  void diag(view1D<vector<Float>, Float> &vec) const;
  void diag(view1D<matrix::Dense<Float>, Float> &vec) const;

  /**
   * @brief operator copy
   * @return copied LinearOperator
   * @note
   * - # of computation: 4 + 2 functions
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void operator=(const LinearOperator<Float> &mat);
};
} // namespace matrix
} // namespace monolish
