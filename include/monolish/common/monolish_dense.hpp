#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include <iostream>
#include <memory>

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
template <typename TYPE, typename Float> class view_Dense;
template <typename TYPE, typename Float> class view_tensor_Dense;

namespace tensor {
template <typename Float> class tensor_Dense;
}

namespace matrix {

/**
 * @addtogroup Dense_class
 * @{
 */

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
  // size_t nnz;

  /**
   * @brief true: sended, false: not send
   */
  mutable std::shared_ptr<bool> gpu_status = std::make_shared<bool>(false);

  /**
   * @brief first position of data array
   */
  size_t first = 0;

public:
  /**
   * @brief Dense format value (pointer)
   */
  std::shared_ptr<Float> val;

  /**
   * @brief # of non-zero element (M * N)
   */
  size_t val_nnz = 0;

  /**
   * @brief alloced matrix size
   */
  size_t alloc_nnz = 0;

  /**
   * @brief matrix create flag;
   */
  bool val_create_flag = false;

  Dense() { val_create_flag = true; }

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
  Dense(const COO<Float> &coo) {
    val_create_flag = true;
    convert(coo);
  }

  /**
   * @brief Create Dense matrix from Dense matrix
   * @param dense Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M*N (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  Dense(const Dense<Float> &dense);

  /**
   * @brief Create Dense matrix of the same size as input matrix
   * @param dense input Dense matrix
   * @param value the value to initialize elements
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M*N (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  Dense(const Dense<Float> &dense, Float value);

  /**
   * @brief Create Dense matrix from view Dense matrix
   * @param dense Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M*N (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  Dense(const view_Dense<vector<Float>, Float> &dense);

  /**
   * @brief Create Dense matrix from view Dense matrix
   * @param dense Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M*N (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  Dense(const view_Dense<matrix::Dense<Float>, Float> &dense);

  /**
   * @brief Create Dense matrix from view Dense matrix
   * @param dense Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M*N (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  Dense(const view_Dense<tensor::tensor_Dense<Float>, Float> &dense);

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
   * @brief Create construct dense matrix
   * @param M # of row
   * @param N # of col
   * @param value value
   * @note
   * - # of computation: 1
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N, const std::shared_ptr<Float> &value);

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
   * The ramdom number generator is random generator is mt19937
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N, const Float min, const Float max);

  /**
   * @brief Create random dense matrix from dense matrix
   * @param M # of row
   * @param N # of col
   * @param min rand min
   * @param max rand max
   * @param seed random seed
   * @note
   * - # of computation: M*N
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  Dense(const size_t M, const size_t N, const Float min, const Float max,
        const std::uint32_t seed);

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
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const size_t M, const size_t N, const std::vector<Float> &value);

  /**
   * @brief Set Dense array from std::vector
   * @param M # of row
   * @param N # of col
   * @param value value (size nnz)
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const size_t M, const size_t N, const Float *value);

  /**
   * @brief Set Dense array from std::vector
   * @param M # of row
   * @param N # of col
   * @param value value (size nnz)
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const size_t M, const size_t N, const Float value);

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
   * @brief get shared_ptr of val
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::shared_ptr<Float> get_val() { return val; }

  /**
   * @brief get # of non-zeros
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_nnz() const { return val_nnz; }

  /**
   * @brief get # of alloced non-zeros
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_alloc_nnz() const { return alloc_nnz; }

  /**
   * @brief get first position
   * @return first position
   * @note
   * - # of computation: 1
   */
  [[nodiscard]] size_t get_first() const { return first; }

  /**
   * @brief get first position (same as get_first())
   * @return first position
   * @note
   * - # of computation: 1
   */
  [[nodiscard]] size_t get_offset() const { return get_first(); }

  /**
   * @brief Set row number
   * @param N # of row
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_row(const size_t N) { rowN = N; };

  /**
   * @brief Set column number
   * @param M # of col
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_col(const size_t M) { colN = M; };

  /**
   * @brief Set # of non-zero elements
   * @param NZ # of non-zero elements
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  // void set_nnz(const size_t NZ) { val_nnz = NZ; };

  /**
   * @brief change first position
   * @note
   * - # of computation: 1
   */
  void set_first(size_t i) { first = i; }

  /**
   * @brief get format name "Dense"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const { return "Dense"; }

  /**
   * @brief get transposed matrix (A = A^T)
   * @note
   * - # of computation: M*N/2-M (square) or M*N (non-square)
   * - Multi-threading: yes
   * - GPU acceleration: false
   * - If matrix is non-square, This function need MxN temporary matrix.
   * - This function transposes in place, it's performance is lower than
   *A.transpose(B).
   **/
  void transpose();

  /**
   * @brief create transposed matrix from Dense format matrix (A = B^T)
   * @param B Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: yes
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
   * @brief get element A[i/col][j%col]
   * @param i index
   * @return A[i/col][j%col]
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t i) const;

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
   * @brief get element A[i/col][i%col] (only CPU)
   * @param i index
   * @return A[i/col][i%col]
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t i) {
    return static_cast<const Dense *>(this)->at(i);
  };

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
   * @brief set element A[i/col][j%col]
   * @param i index
   * @param Val scalar value
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void insert(const size_t i, const Float Val);

  /**
   * @brief set element A[i][j]
   * @param i row
   * @param j col
   * @param Val scalar value
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
  [[nodiscard]] bool get_device_mem_stat() const { return *gpu_status; }

  /**
   * @brief gpu status shared pointer
   * @return gpu status shared pointer
   */
  [[nodiscard]] std::shared_ptr<bool> get_gpu_status() const {
    return gpu_status;
  }

  /**
   * @brief destructor of Dense matrix, free GPU memory
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: 0
   * **/
  ~Dense() {
    if (val_create_flag) {
      if (get_device_mem_stat()) {
        device_free();
      }
    }
  }

  /**
   * @brief returns a direct pointer to the matrix
   * @return A const pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *data() const { return val.get(); }

  /**
   * @brief returns a direct pointer to the matrix
   * @return A pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *data() { return val.get(); }

  /**
   * @brief resize matrix value
   * @param N matrix size
   * @note
   * - # of computation: N
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void resize(size_t N, Float Val = 0) {
    if (first + N < alloc_nnz) {
      for (size_t i = val_nnz; i < N; ++i) {
        begin()[i] = Val;
      }
      val_nnz = N;
      return;
    }
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU matrix cant use resize");
    }
    if (val_create_flag) {
      std::shared_ptr<Float> tmp(new Float[N], std::default_delete<Float[]>());
      size_t copy_size = std::min(val_nnz, N);
      for (size_t i = 0; i < copy_size; ++i) {
        tmp.get()[i] = data()[i];
      }
      for (size_t i = copy_size; i < N; ++i) {
        tmp.get()[i] = Val;
      }
      val = tmp;
      alloc_nnz = N;
      val_nnz = N;
      first = 0;
    } else {
      throw std::runtime_error("Error, not create vector cant use resize");
    }
  }

  void move(const tensor::tensor_Dense<Float> &tensor_dense);

  void move(const tensor::tensor_Dense<Float> &tensor_dense, int rowN,
            int colN);

  void move(const view_tensor_Dense<vector<Float>, Float> &tensor_dense);

  void move(const view_tensor_Dense<matrix::Dense<Float>, Float> &tensor_dense);

  void move(const view_tensor_Dense<tensor::tensor_Dense<Float>, Float>
                &tensor_dense);

  void move(const view_tensor_Dense<vector<Float>, Float> &tensor_dense,
            int rowN, int colN);

  void move(const view_tensor_Dense<matrix::Dense<Float>, Float> &tensor_dense,
            int rowN, int colN);

  void move(
      const view_tensor_Dense<tensor::tensor_Dense<Float>, Float> &tensor_dense,
      int rowN, int colN);

  /**
   * @brief returns a begin iterator
   * @return begin iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *begin() const { return data() + get_offset(); }

  /**
   * @brief returns a begin iterator
   * @return begin iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *begin() { return data() + get_offset(); }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *end() const {
    return data() + get_offset() + get_nnz();
  }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *end() { return data() + get_offset() + get_nnz(); }

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
  void diag(view1D<tensor::tensor_Dense<Float>, Float> &vec) const;

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
  void row(const size_t r,
           view1D<tensor::tensor_Dense<Float>, Float> &vec) const;

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
  void col(const size_t c,
           view1D<tensor::tensor_Dense<Float>, Float> &vec) const;

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
   * @param mat Dense matrix
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
   * @brief matrix copy
   * @param mat Dense matrix
   * @return copied dense matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: 0
   *        - if `gpu_statius == true`; coping data on CPU
   *        - else; coping data on CPU
   **/
  void operator=(const view_Dense<vector<Float>, Float> &mat);

  /**
   * @brief matrix copy
   * @param mat Dense matrix
   * @return copied dense matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: 0
   *        - if `gpu_statius == true`; coping data on CPU
   *        - else; coping data on CPU
   **/
  void operator=(const view_Dense<matrix::Dense<Float>, Float> &mat);

  /**
   * @brief matrix copy
   * @param mat Dense matrix
   * @return copied dense matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: 0
   *        - if `gpu_statius == true`; coping data on CPU
   *        - else; coping data on CPU
   **/
  void operator=(const view_Dense<tensor::tensor_Dense<Float>, Float> &mat);

  /**
   * @brief reference to the element at position (v[i])
   * @param i Position of an element in the vector
   * @return vector element (v[i])
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float &operator[](size_t i) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    return data()[first + i];
  }

  /**
   * @brief Comparing matrices (A == mat)
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
   * @brief Comparing matrices (A == mat)
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
   * @brief Comparing matrices (A != mat)
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

  /**
   * @brief Reshape matrix
   * @param row
   * @param col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void reshape(const size_t row, const size_t col);

  /////////////////////////////////////////////

  /**
   * @brief Scalar and diag. vector of Dense format matrix add
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_add(const Float alpha);

  /**
   * @brief Scalar and diag. vector of Dense format matrix sub
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_sub(const Float alpha);

  /**
   * @brief Scalar and diag. vector of Dense format matrix mul
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_mul(const Float alpha);

  /**
   * @brief Scalar and diag. vector of Dense format matrix div
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_div(const Float alpha);

  /**
   * @brief Vector and diag. vector of Dense format matrix add
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_add(const vector<Float> &vec);
  void diag_add(const view1D<vector<Float>, Float> &vec);
  void diag_add(const view1D<matrix::Dense<Float>, Float> &vec);
  void diag_add(const view1D<tensor::tensor_Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of Dense format matrix sub
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_sub(const vector<Float> &vec);
  void diag_sub(const view1D<vector<Float>, Float> &vec);
  void diag_sub(const view1D<matrix::Dense<Float>, Float> &vec);
  void diag_sub(const view1D<tensor::tensor_Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of Dense format matrix mul
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_mul(const vector<Float> &vec);
  void diag_mul(const view1D<vector<Float>, Float> &vec);
  void diag_mul(const view1D<matrix::Dense<Float>, Float> &vec);
  void diag_mul(const view1D<tensor::tensor_Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of Dense format matrix div
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_div(const vector<Float> &vec);
  void diag_div(const view1D<vector<Float>, Float> &vec);
  void diag_div(const view1D<matrix::Dense<Float>, Float> &vec);
  void diag_div(const view1D<tensor::tensor_Dense<Float>, Float> &vec);
};
/**@}*/

} // namespace matrix
} // namespace monolish
