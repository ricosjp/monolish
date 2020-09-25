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
namespace matrix {
/**
 * @brief Dense format Matrix
 * @note
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
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
   * @brief # of non-zero element
   */
  size_t nnz;

  /**
   * @brief true: sended, false: not send
   */
  mutable bool gpu_status = false; // true: sended, false: not send

  /**
   * @brief Set row number
   **/
  void set_row(const size_t N) { rowN = N; };

  /**
   * @brief Set column number
   **/
  void set_col(const size_t M) { colN = M; };

  /**
   * @brief Set # of non-zero elements
   **/
  void set_nnz(const size_t NZ) { nnz = NZ; };

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
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): false
   **/
  void convert(const COO<Float> &coo);

  /**
   * @brief Create dense matrix from COO matrix
   * @param coo input COO matrix (size M x N)
   * @note
   * - # of computation: M*N + nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): false
   **/
  Dense(const COO<Float> &coo) { convert(coo); }

  /**
   * @brief Create dense matrix from dense matrix
   * @param mat input dense matrix (size M x N)
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   **/
  Dense(const Dense<Float> &mat);

  /**
   * @brief Allocate dense matrix
   * @param M # of row
   * @param N # of col
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  Dense(const size_t M, const size_t N) {
    set_row(M);
    set_col(N);
    set_nnz(N * M);

    val.resize(nnz);
  }

  /**
   * @brief Create dense matrix from array
   * @param M # of row
   * @param N # of col
   * @param value value array
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  Dense(const size_t M, const size_t N, const Float *value) {
    set_row(M);
    set_col(N);
    set_nnz(M * N);

    val.resize(nnz);
    std::copy(value, value + nnz, val.begin());
  }

  /**
   * @brief Create dense matrix from std::vector
   * @param M # of row
   * @param N # of col
   * @param value value std::vector (size M x N)
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  Dense(const size_t M, const size_t N, const std::vector<Float> value) {
    set_row(M);
    set_col(N);
    set_nnz(M * N);

    val.resize(nnz);
    std::copy(value.data(), value.data() + nnz, val.begin());
  }

  /**
   * @brief Create random dense matrix from dense matrix
   * @param M # of row
   * @param N # of col
   * @param min rand min
   * @param max rand max
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): false
   **/
  Dense(const size_t M, const size_t N, const Float min, const Float max) {
    set_row(M);
    set_col(N);
    set_nnz(M * N);

    val.resize(nnz);

    std::random_device random;
    std::mt19937 mt(random());
    std::uniform_real_distribution<> rand(min, max);

#pragma omp parallel for
    for (size_t i = 0; i < val.size(); i++) {
      val[i] = rand(mt);
    }
  }

  /**
   * @brief Create constract dense matrix
   * @param M # of row
   * @param N # of col
   * @param value value
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): false
   **/
  Dense(const size_t M, const size_t N, const Float value) {
    set_row(M);
    set_col(N);
    set_nnz(M * N);

    val.resize(nnz);

#pragma omp parallel for
    for (size_t i = 0; i < val.size(); i++) {
      val[i] = value;
    }
  }

  /**
   * @brief get # of row
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  size_t get_row() const { return rowN; }

  /**
   * @brief get # of col
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  size_t get_col() const { return colN; }

  /**
   * @brief get # of nnz
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  size_t get_nnz() const { return get_row() * get_col(); }

  /**
   * @brief get format name "Dense"
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::string type() const { return "Dense"; }

  /*
   * @brief get transposed matrix (A^T)
   * @return tranposed matrix A^T
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   * @warning
   * This function need to allocate tmp. matrix (size M x N)
   **/
  Dense &transpose() {
    Dense<Float> B(get_col(), get_row());
    for (size_t i = 0; i < get_row(); ++i) {
      for (size_t j = 0; j < get_col(); ++j) {
        B.val[j * get_row() + i] = val[i * get_col() + j];
      }
    }
    std::copy(B.val.data(), B.val.data() + nnz, val.begin());
    set_row(B.get_row());
    set_col(B.get_col());
    return *this;
  }

  /*
   * @brief create transposed matrix from COO matrix (A = B^T)
   * @param B COO matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void transpose(const Dense &B) {
    set_row(B.get_col());
    set_col(B.get_row());
    val.resize(B.get_row() * B.get_col());

    for (size_t i = 0; i < get_row(); ++i) {
      for (size_t j = 0; j < get_col(); ++j) {
        val[j * get_row() + i] = B.val[i * get_col() + j];
      }
    }
  }

  /*
   * @brief Memory data space required by the matrix
   * @note
   * - # of computation: 3
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  double get_data_size() const { return get_nnz() * sizeof(Float) / 1.0e+9; }

  /**
   * @brief get element A[i][j] (only CPU)
   * @param i row
   * @param j col
   * @return A[i][j]
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  Float at(size_t i, size_t j) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("at() Error, GPU vector cant use operator[]");
    }
    if (get_row() < i) {
      throw std::runtime_error("at() Error, A.row < i");
    }
    if (get_col() < j) {
      throw std::runtime_error("at() Error, A.col < j");
    }
    return val[get_col() * i + j];
  }

  /**
   * @brief get element A[i][j]
   * @param i row
   * @param j col
   * @return A[i][j]
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  Float at(size_t i, size_t j) const {
    if (get_device_mem_stat()) {
      throw std::runtime_error("at() Error, GPU vector cant use operator[]");
    }
    if (get_row() < i) {
      throw std::runtime_error("at() Error, A.row < i");
    }
    if (get_col() < j) {
      throw std::runtime_error("at() Error, A.col < j");
    }
    return val[get_col() * i + j];
  }

  /**
   * @brief get element A[i][j]
   * @param i row
   * @param j col
   * @param Val scalar value
   * @return A[i][j]
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void insert(size_t i, size_t j, Float Val) {
    if (get_device_mem_stat()) {
      throw std::runtime_error(
          "insert() Error, GPU vector cant use operator[]");
    }
    if (get_row() < i) {
      throw std::runtime_error("insert() Error, A.row < i");
    }
    if (get_col() < j) {
      throw std::runtime_error("insert() Error, A.col < j");
    }
    val[get_col() * i + j] = Val;
  }

  /**
   * @brief print all elements to standart I/O
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void print_all();

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: M*N
   **/
  void send() const;

  /**
   * @brief recv. data to GPU, and free data on GPU
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: M*N
   **/
  void recv();

  /**
   * @brief recv. data to GPU (w/o free)
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: M*N
   **/
  void nonfree_recv();

  /**
   * @brief free data on GPU
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: 0
   **/
  void device_free() const;

  /**
   * @brief true: sended, false: not send
   * @return gpu status
   * **/
  bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief destructor of CRS matrix, free GPU memory
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: 0
   * **/
  ~Dense() {
    //       if(get_device_mem_stat()){
    //           device_free();
    //       }
  }

  /////////////////////////////////////////////////////////////////////////////
  /**
   * @brief get diag. vector
   * @param vec diag. vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag(vector<Float> &vec) const;

  /**
   * @brief get row vector
   * @param r row number
   * @param vec row vector
   * @note
   * - # of computation: about nnz / M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row(const size_t r, vector<Float> &vec) const;

  /**
   * @brief get column vector
   * @param c column number
   * @param vec column vector
   * @note
   * - # of computation: about nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col(const size_t c, vector<Float> &vec) const;

  /////////////////////////////////////////////////////////////////////////////

  /**
   * @brief matrix copy
   * @return copied dense matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: M*N
   *        - if `vec.gpu_statius == true`; copy on CPU; then send to GPU
   *        - else; coping data only on CPU
   **/
  Dense copy();

  /**
   * @brief matrix copy
   * @return copied dense matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: M*N
   *        - if `vec.gpu_statius == true`; copy on CPU; then send to GPU
   *        - else; coping data only on CPU
   **/
  void operator=(const Dense<Float> &mat);

  /**
   * @brief matrix scale (value*A)
   * @param value scalar value
   * @return CRS matrix (value*A)
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  Dense<Float> operator*(const Float value);

  /**
   * @brief matrix-vector multiplication (A*vec)
   * @param vec vector (size N)
   * @return result vector (size M)
   * @note
   * - # of computation: 2*M*N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  vector<Float> operator*(vector<Float> &vec);

  /**
   * @brief Dense matrix (size M*K) and dense matrix (size K*N) multiplication
   *(A*B)
   * @param B Dense matrix (size K*N)
   * @return result dense matrix (size M*N)
   * @note
   * - # of computation: 2*M*N*K
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  Dense<Float> operator*(const Dense<Float> &B);

  /**
   * @brief Dense matrix (size M*N) and dense matrix (size K*N) addition (A+B)
   * @param B CRS matrix (size M*N)
   * @return result CRS matrix (size M*N)
   * @note
   * - # of computation: M*N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  Dense<Float> operator+(const Dense<Float> &B);

  /////////////////////////////////////////////

  /**
   * @brief Scalar and diag. vector of dense matrix add
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag_add(const Float alpha);

  /**
   * @brief Scalar and diag. vector of dense matrix sub
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag_sub(const Float alpha);

  /**
   * @brief Scalar and diag. vector of dense matrix mul
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag_mul(const Float alpha);

  /**
   * @brief Scalar and diag. vector of dense matrix div
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag_div(const Float alpha);

  /**
   * @brief Vector and diag. vector of dense matrix add
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag_add(const vector<Float> &vec);

  /**
   * @brief Vector and diag. vector of dense matrix sub
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag_sub(const vector<Float> &vec);

  /**
   * @brief Vector and diag. vector of dense matrix mul
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag_mul(const vector<Float> &vec);

  /**
   * @brief Vector and diag. vector of dense matrix div
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void diag_div(const vector<Float> &vec);

  ///////////////////////////////////////////////////////////////

  /**
   * @brief Scalar and row vector of dense matrix add
   * @param r row number
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row_add(const size_t r, const Float alpha);

  /**
   * @brief Scalar and row vector of dense matrix sub
   * @param r row number
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row_sub(const size_t r, const Float alpha);

  /**
   * @brief Scalar and row vector of dense matrix mul
   * @param r row number
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row_mul(const size_t r, const Float alpha);

  /**
   * @brief Scalar and row vector of dense matrix div
   * @param r row number
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row_div(const size_t r, const Float alpha);

  /**
   * @brief Vector and row vector of dense matrix add
   * @param r row number
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row_add(const size_t r, const vector<Float> &vec);

  /**
   * @brief Vector and row vector of dense matrix sub
   * @param r row number
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row_mul(const size_t r, const vector<Float> &vec);

  /**
   * @brief Vector and row vector of dense matrix mul
   * @param r row number
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row_sub(const size_t r, const vector<Float> &vec);

  /**
   * @brief Vector and row vector of dense matrix div
   * @param r row number
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void row_div(const size_t r, const vector<Float> &vec);

  ///////////////////////////////////////////////////////////////

  /**
   * @brief Scalar and column vector of dense matrix add
   * @param c column number
   * @param alpha scalar
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col_add(const size_t c, const Float alpha);

  /**
   * @brief Scalar and column vector of dense matrix sub
   * @param c column number
   * @param alpha scalar
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col_sub(const size_t c, const Float alpha);

  /**
   * @brief Scalar and column vector of dense matrix mul
   * @param c column number
   * @param alpha scalar
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col_mul(const size_t c, const Float alpha);

  /**
   * @brief Scalar and column vector of dense matrix div
   * @param c column number
   * @param alpha scalar
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col_div(const size_t c, const Float alpha);

  /**
   * @brief Vector and column vector of dense matrix add
   * @param c column number
   * @param vec vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col_add(const size_t c, const vector<Float> &vec);

  /**
   * @brief Vector and column vector of dense matrix sub
   * @param c column number
   * @param vec vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col_mul(const size_t c, const vector<Float> &vec);

  /**
   * @brief Vector and column vector of dense matrix mul
   * @param c column number
   * @param vec vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col_sub(const size_t c, const vector<Float> &vec);

  /**
   * @brief Vector and column vector of dense matrix div
   * @param c column number
   * @param vec vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void col_div(const size_t c, const vector<Float> &vec);
};
} // namespace matrix
} // namespace monolish
