/**
 * @author fockl
 * @file monolish_linearoperator.h
 * @brief declare linearoperator class
 * @date 2021
 **/

#pragma once
#include <exception>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

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
   * @brief flag that shows matvec is defined or not
   */
  bool matvec_init_flag;

  /**
   * @brief flag that shows rmatvec is defined or not
   */
  bool rmatvec_init_flag;

  /**
   * @brief pseudo multiplication function of matrix and vector
   */
  std::function<vector<Float>(const vector<Float>&)> matvec;

  /**
   * @brief pseudo multiplication function of (Hermitian) transposed matrix and vector
   */
  std::function<vector<Float>(const vector<Float>&)> rmatvec;

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
  LinearOperator(const size_t M, const size_t N, const std::function<vector<Float>(const vector<Float>&)>& MATVEC);

  /**
   * @brief declare LinearOperator
   * @param M # of row
   * @param N # of col
   * @param MATVEC multiplication function of matrix and vector
   * @param RMATVEC multiplication function of (Hermitian) transposed matrix and vector
   * @note
   * - # of computation: 4 + 2 functions
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  LinearOperator(const size_t M, const size_t N, const std::function<vector<Float>(const vector<Float>&)>& MATVEC, const std::function<vector<Float>(const vector<Float>&)>& RMATVEC);

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

  void convert_to_Dense(Dense<Float> &dense) const ;

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
  size_t get_row() const { return rowN; }

  /**
   * @brief get # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  size_t get_col() const { return colN; }

  /**
   * @brief get multiplication function of matrix and vector
   * @note
   * - # of computation: 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  std::function<vector<Float>(const vector<Float>&)> get_matvec() const { return matvec; }

  /**
   * @brief get multiplication function of (Hermitian) transposed matrix and vector
   * @note
   * - # of computation: 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  std::function<vector<Float>(const vector<Float>&)> get_rmatvec() const { return rmatvec; }

  /**
   * @brief get flag that shows matvec is defined or not
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  bool get_matvec_init_flag() const { return matvec_init_flag; }

  /**
   * @brief get flag that shows rmatvec is defined or not
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  bool get_rmatvec_init_flag() const { return rmatvec_init_flag; }

  /**
   * @brief set multiplication function of matrix and vector
   * @note
   * - # of computation: 1 + 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void set_matvec(const std::function<vector<Float>(const vector<Float>&)>& MATVEC);

  /**
   * @brief set multiplication function of (Hermitian) transposed matrix and vector
   * @note
   * - # of computation: 1 + 1 function
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void set_rmatvec(const std::function<vector<Float>(const vector<Float>&)>& RMATVEC);

  /**
   * @brief get format name "LinearOperator"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string type() const { return "LinearOperator"; }

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
  bool get_device_mem_stat() const {return false;};

  /**
   * @brief destructor of LinearOperator, free GPU memory
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   * **/
  ~LinearOperator() {}

  /**
   * @brief operator copy
   * @return copied LinearOperator
   * @note
   * - # of computation: 4 + 2 functions
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  LinearOperator copy();

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
} // namespace linearoperator
} // namespace monolish
