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
 * - Multi-threading: true
 * - GPU acceleration: true
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
   * @brief declare CRS matrix
   * @param M # of row
   * @param N # of col
   * @param NNZ # of nnz
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  CRS(const size_t M, const size_t N, const size_t NNZ);

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
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
      const int *colind, const Float *value);

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
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  CRS(const size_t M, const size_t N, const std::vector<int> rowptr,
      const std::vector<int> colind, const std::vector<Float> value);

  /**
   * @brief Convert CRS from COO
   * @param coo COO format matrix
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void convert(COO<Float> &coo);

  /**
   * @brief Create CRS from COO
   * @param coo Source COO format matrix
   * @return coo COO format matrix
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  CRS(COO<Float> &coo) { convert(coo); }

  /**
   * @brief Create CRS from CRS
   * @param mat CRS format matrix
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: (M+1)+2nnz (allocation)
   *        - if `vec.gpu_statius == true`; copy on CPU; then send to GPU
   *        - else; coping data only on CPU
   **/
  CRS(const CRS<Float> &mat);

  /**
   * @brief print all elements to standard I/O
   * @param force_cpu Ignore device status and output CPU data
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void print_all(bool force_cpu=false) const;

  /**
   * @brief get # of row
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  size_t get_row() const { return rowN; }

  /**
   * @brief get # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  size_t get_col() const { return colN; }

  /**
   * @brief get # of non-zeros
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  size_t get_nnz() const { return nnz; }

  /**
   * @brief get format name "CRS"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string type() const { return "CRS"; }

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: (M+1) + 2nnz
   **/
  void send() const;

  /**
   * @brief recv. data to GPU, and free data on GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: (M+1) + 2nnz
   **/
  void recv();

  /**
   * @brief recv. data to GPU (w/o free)
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: (M+1) + 2nnz
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
  bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief destructor of CRS matrix, free GPU memory
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
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
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag(vector<Float> &vec) const;

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

  /////////////////////////////////////////////////////////////////////////////

  /*
   * @brief Memory data space required by the matrix
   * @note
   * - # of computation: 3
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  double get_data_size() const {
    return (get_nnz() * sizeof(Float) + (get_row() + 1) * sizeof(int) +
            get_nnz() * sizeof(int)) /
           1.0e+9;
  }

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
   * @return copied CRS matrix
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: (M+1)+2nnz (allocation)
   *        - if `vec.gpu_statius == true`; copy on CPU; then send to GPU
   *        - else; coping data only on CPU
   **/
  CRS copy();

  /**
   * @brief matrix copy
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: (M+1)+2nnz (allocation)
   *        - if `vec.gpu_statius == true`; copy on GPU
   *        - else; coping data only on CPU
   **/
  void operator=(const CRS<Float> &mat);

  /**
   * @brief Comparing matricies (A == mat)
   * @param mat CRS matrix
   * @return true or false
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  bool operator==(const CRS<Float> &mat) const;

  /**
   * @brief Comparing matricies (A != mat)
   * @param mat CRS matrix
   * @return true or false
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  bool operator!=(const CRS<Float> &mat) const;
};
} // namespace matrix
} // namespace monolish
