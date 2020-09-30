/**
 * @author RICOS Co. Ltd.
 * @file monolish_vector.h
 * @brief declare vector class
 * @date 2019
 **/

#pragma once
#include <exception>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace monolish {
template <typename Float> class vector;
namespace matrix {
template <typename Float> class Dense;
template <typename Float> class COO;

/**
 * @brief Compressed Row Storage (CRS) format Matrix
 * @note
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
template <typename Float> class CRS {
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
  mutable bool gpu_status = false;

public:
  /**
   * @brief CRS format value, which stores values of the non-zero elements (size
   * nnz)
   */
  std::vector<Float> val;

  /**
   * @brief CRS format column index, which stores column numbers of the non-zero
   * elements (size nnz)
   */
  std::vector<int> col_ind;

  /**
   * @brief CRS format row pointer, which stores the starting points of the rows
   * of the arrays value and col_ind (size M+1)
   */
  std::vector<int> row_ptr;

  CRS() {}

  /**
   * @brief Create CRS matrix from array
   * @param M # of row
   * @param N # of col
   * @param NNZ # of non-zero elements
   * @param rowptr row_ptr, which stores the starting points of the rows of the
   *arrays value and col_ind (size M+1)
   * @param colind col_ind, which stores the column numbers of the non-zero
   *elements (size nnz)
   * @param value value index, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
      const int *colind, const Float *value)
      : rowN(M), colN(N), nnz(NNZ), gpu_status(false), row_ptr(M + 1),
        col_ind(nnz), val(nnz) {
    std::copy(rowptr, rowptr + (M + 1), row_ptr.begin());
    std::copy(colind, colind + nnz, col_ind.begin());
    std::copy(value, value + nnz, val.begin());
  }

  /**
   * @brief Create CRS matrix from std::vector
   * @param M # of row
   * @param N # of col
   * @param rowptr row_ptr, which stores the starting points of the rows of the
   *arrays value and col_ind (size M+1)
   * @param colind col_ind, which stores the column numbers of the non-zero
   *elements (size nnz)
   * @param value value index, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  CRS(const size_t M, const size_t N, const std::vector<int> rowptr,
      const std::vector<int> colind, const std::vector<Float> value)
      : rowN(M), colN(N), nnz(value.size()), gpu_status(false), row_ptr(M + 1),
        col_ind(nnz), val(nnz) {
    std::copy(rowptr.data(), rowptr.data() + (M + 1), row_ptr.begin());
    std::copy(colind.data(), colind.data() + nnz, col_ind.begin());
    std::copy(value.data(), value.data() + nnz, val.begin());
  }

  /**
   * @brief Convert CRS from COO
   * @param coo COO format matrix
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void convert(COO<Float> &coo);

  /**
   * @brief Create CRS from COO
   * @param coo Source COO format matrix
   * @return coo COO format matrix
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  CRS(COO<Float> &coo) { convert(coo); }

  /**
   * @brief Create CRS from CRS
   * @param mat CRS format matrix
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: (M+1)+2nnz (allocation)
   *        - if `vec.gpu_statius == true`; copy on CPU; then send to GPU
   *        - else; coping data only on CPU
   **/
  CRS(const CRS<Float> &mat);

  /**
   * @brief print all elements to standart I/O
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void print_all();

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
   * @brief matrix copy
   * @return copied COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  size_t get_nnz() const { return nnz; }

  /**
   * @brief get format name "CRS"
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::string type() const { return "CRS"; }

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: (M+1) + 2nnz
   **/
  void send() const;

  /**
   * @brief recv. data to GPU, and free data on GPU
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: (M+1) + 2nnz
   **/
  void recv();

  /**
   * @brief recv. data to GPU (w/o free)
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: (M+1) + 2nnz
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
  ~CRS() {
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

  /*
   * @brief Memory data space required by the matrix
   * @note
   * - # of computation: 3
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  double get_data_size() const {
    return (get_nnz() * sizeof(Float) + (get_row() + 1) * sizeof(int) +
            get_nnz() * sizeof(int)) /
           1.0e+9;
  }

  /**
   * @brief matrix copy
   * @return copied CRS matrix
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: (M+1)+2nnz (allocation)
   *        - if `vec.gpu_statius == true`; copy on CPU; then send to GPU
   *        - else; coping data only on CPU
   **/
  CRS copy();

  /**
   * @brief matrix copy
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: (M+1)+2nnz (allocation)
   *        - if `vec.gpu_statius == true`; copy on GPU
   *        - else; coping data only on CPU
   **/
  void operator=(const CRS<Float> &mat);

  /**
   * @brief matrix scale (value*A)
   * @param value scalar value
   * @return CRS matrix (value*A)
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  CRS<Float> operator*(const Float value);

  /**
   * @brief matrix-vector multiplication (A*vec)
   * @param vec vector (size N)
   * @return result vector (size M)
   * @note
   * - # of computation: 2nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  vector<Float> operator*(vector<Float> &vec);

  /**
   * @brief CRS matrix (size M*K) and Dense matrix (size K*N) multiplication
   *(A*B)
   * @param B Dense matrix (size K*N)
   * @return result Dense matrix (size M*N)
   * @note
   * - # of computation: 2*N*nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  Dense<Float> operator*(const Dense<Float> &B);

  // crs-dense
  /**
   * @brief CRS matrix (size M*N) and CRS matrix (size K*N) addition A + B (A and B must be same non-zero structure)
   * @param B CRS matrix (size M*N)
   * @return result CRS matrix (size M*N)
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  CRS<Float> operator+(const CRS<Float> &B);

  /**
   * @brief tanh vector elements (A(i,j) = tanh(A(0:j)))
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void tanh();
};
} // namespace matrix
} // namespace monolish
