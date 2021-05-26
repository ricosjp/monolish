/**
 * @autor RICOS Co. Ltd.
 * @file monolish_vector.h
 * @brief declare vector class
 * @date 2019
 **/

#pragma once
#include "monolish_matrix.hpp"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace matrix {
/**
 * @brief Dense format Matrix
 * @note
 * - Multi-threading: true
 * - GPU acceleration: true
 */
template <typename Float> class Dense {
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
   * @brief # of non-zero element (M * N)
   */
  size_t nnz;

  /**
   * @brief true: sended, false: not send
   */
  mutable bool gpu_status = false;

public:
  /**
   * @brief Dense format value(size M x N)
   */
  std::vector<Float> val;

  Dense() {}

  /**
   * @brief Create Dense matrix from COO matrix
   * @param coo input COO matrix (size M x N)
   * @note
   * - # of computation: M*N + nnz
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const COO<Float> &coo);

  /**
   * @brief Create Dense matrix from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const Dense<Float> &dense);

  /**
   * @brief Create dense matrix from COO matrix
   * @param coo input COO matrix (size M x N)
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  Dense(const COO<Float> &coo) { convert(coo); }

  /**
   * @brief Create Dense matrix from Dense matrix
   * @param dense Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M+N (onlu allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  Dense(const Dense<Float> &dense);

  /**
   * @brief Allocate dense matrix
   * @param M # of row
   * @param N # of col
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N);

  /**
   * @brief Create dense matrix from array
   * @param M # of row
   * @param N # of col
   * @param value value array
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N, const Float *value);

  /**
   * @brief Create dense matrix from std::vector
   * @param M # of row
   * @param N # of col
   * @param value value std::vector (size M x N)
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N, const std::vector<Float> &value);

  /**
   * @brief Create dense matrix from monolish::vector
   * @param M # of row
   * @param N # of col
   * @param value value std::vector (size M x N)
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: true
   **/
  Dense(const size_t M, const size_t N, const vector<Float> &value);

  /**
   * @brief Create dense matrix from std::initializer_list
   * @param M # of row
   * @param N # of col
   * @param list value std::initializer_list (size M x N)
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N,
        const std::initializer_list<Float> &list);

  /**
   * @brief Create random dense matrix from dense matrix
   * @param M # of row
   * @param N # of col
   * @param min rand min
   * @param max rand max
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N, const Float min, const Float max);

  /**
   * @brief Create construct dense matrix
   * @param M # of row
   * @param N # of col
   * @param value value
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N, const Float value);

  /**
   * @brief Set Dense array from std::vector
   * @param M # of row
   * @param N # of col
   * @param value value (size nnz)
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const size_t M, const size_t N, const std::vector<Float> &value);

  /**
   * @brief get # of row
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_row() const { return rowN; }

  /**
   * @brief get # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_col() const { return colN; }

  /**
   * @brief get # of non-zeros
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_nnz() const { return get_row() * get_col(); }

  /**
   * @brief Set row number
   * @param M # of row
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_row(const size_t N) { rowN = N; };

  /**
   * @brief Set column number
   * @param N # of col
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_col(const size_t M) { colN = M; };

  /**
   * @brief Set # of non-zero elements
   * @param NNZ # of non-zero elements
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_nnz(const size_t NZ) { nnz = NZ; };

  /**
   * @brief get format name "Dense"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const { return "Dense"; }

  /**
   * @brief get transposed matrix (A^T)
   * @return tranposed matrix A^T
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: false
   * @warning
   * This function need to allocate tmp. matrix (size M x N)
   **/
  Dense &transpose();

  /**
   * @brief create transposed matrix from COO matrix (A = B^T)
   * @param B COO matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void transpose(const Dense &B);

  /**
   * @brief Memory data space required by the matrix
   * @note
   * - # of computation: 3
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] double get_data_size() const {
    return get_nnz() * sizeof(Float) / 1.0e+9;
  }

  /**
   * @brief get element A[i][j]
   * @param i row
   * @param j col
   * @return A[i][j]
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t i, const size_t j) const;

  /**
   * @brief get element A[i][j] (only CPU)
   * @param i row
   * @param j col
   * @return A[i][j]
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t i, const size_t j) {
    return static_cast<const Dense *>(this)->at(i, j);
  };

  /**
   * @brief get element A[i][j]
   * @param i row
   * @param j col
   * @param Val scalar value
   * @return A[i][j]
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void insert(const size_t i, const size_t j, const Float Val);

  /**
   * @brief print all elements to standard I/O
   * @param force_cpu Ignore device status and output CPU data
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void print_all(bool force_cpu = false) const;

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: M*N
   **/
  void send() const;

  /**
   * @brief recv. data to GPU, and free data on GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: M*N
   **/
  void recv();

  /**
   * @brief recv. data to GPU (w/o free)
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: M*N
   **/
  void nonfree_recv();

  /**
   * @brief free data on GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: 0
   **/
  void device_free() const;

  /**
   * @brief true: sended, false: not send
   * @return gpu status
   * **/
  [[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief destructor of CRS matrix, free GPU memory
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: 0
   * **/
  ~Dense() {
    if (get_device_mem_stat()) {
      device_free();
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  /**
   * @brief get diag. vector
   * @param vec diag. vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag(vector<Float> &vec) const;
  void diag(view1D<vector<Float>, Float> &vec) const;
  void diag(view1D<matrix::Dense<Float>, Float> &vec) const;

  /**
   * @brief get row vector
   * @param r row number
   * @param vec row vector
   * @note
   * - # of computation: about nnz / M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void row(const size_t r, vector<Float> &vec) const;
  void row(const size_t r, view1D<vector<Float>, Float> &vec) const;
  void row(const size_t r, view1D<matrix::Dense<Float>, Float> &vec) const;

  /**
   * @brief get column vector
   * @param c column number
   * @param vec column vector
   * @note
   * - # of computation: about nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void col(const size_t c, vector<Float> &vec) const;
  void col(const size_t c, view1D<vector<Float>, Float> &vec) const;
  void col(const size_t c, view1D<matrix::Dense<Float>, Float> &vec) const;

  /////////////////////////////////////////////////////////////////////////////

  /**
   * @brief fill matrix elements with a scalar value
   * @param value scalar value
   * @note
   * - # of computation: N
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void fill(Float value);

  /**
   * @brief matrix copy
   * @return copied dense matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: 0
   *        - if `gpu_statius == true`; coping data on CPU
   *        - else; coping data on CPU
   **/
  void operator=(const Dense<Float> &mat);

  /**
   * @brief reference to the pointer of the begining of the m-th row
   * @param m Position of an pointer in the matrix
   * @return pointer at the begining of m-th row
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   * @warning
   * This function is only available for Dense.
   **/
  [[nodiscard]] Float *operator[](size_t m) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU matrix dense cant use operator[]");
    }
    return val.data() + m * get_col();
  }

  /**
   * @brief Comparing matricies (A == mat)
   * @param mat Dense matrix
   * @param compare_cpu_and_device compare data on both CPU and GPU
   * @return true or false
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  [[nodiscard]] bool equal(const Dense<Float> &mat,
                           bool compare_cpu_and_device = false) const;

  /**
   * @brief Comparing matricies (A == mat)
   * @param mat Dense matrix
   * @return true or false
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator==(const Dense<Float> &mat) const;

  /**
   * @brief Comparing matricies (A != mat)
   * @param mat Dense matrix
   * @return true or false
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator!=(const Dense<Float> &mat) const;

  /////////////////////////////////////////////

  /**
   * @brief Scalar and diag. vector of dense matrix add
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_add(const Float alpha);

  /**
   * @brief Scalar and diag. vector of dense matrix sub
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_sub(const Float alpha);

  /**
   * @brief Scalar and diag. vector of dense matrix mul
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_mul(const Float alpha);

  /**
   * @brief Scalar and diag. vector of dense matrix div
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_div(const Float alpha);

  /**
   * @brief Vector and diag. vector of dense matrix add
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_add(const vector<Float> &vec);
  void diag_add(const view1D<vector<Float>, Float> &vec);
  void diag_add(const view1D<matrix::Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of dense matrix sub
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_sub(const vector<Float> &vec);
  void diag_sub(const view1D<vector<Float>, Float> &vec);
  void diag_sub(const view1D<matrix::Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of dense matrix mul
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_mul(const vector<Float> &vec);
  void diag_mul(const view1D<vector<Float>, Float> &vec);
  void diag_mul(const view1D<matrix::Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of dense matrix div
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_div(const vector<Float> &vec);
  void diag_div(const view1D<vector<Float>, Float> &vec);
  void diag_div(const view1D<matrix::Dense<Float>, Float> &vec);
};
} // namespace matrix
} // namespace monolish
