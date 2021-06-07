/**
 * @author RICOS Co. Ltd.
 * @file monolish_vector.h
 * @brief declare vector class
 * @date 2019
 **/

#pragma once
#include <exception>
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
template <typename TYPE, typename Float> class view1D;
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

  /**
   * @brief hash, created from row_ptr and col_ind
   */
  size_t structure_hash;

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
   * @brief Create CRS matrix from array, also compute the hash
   * @param M # of row
   * @param N # of col
   * @param NNZ # of non-zero elements
   * @param rowptr row_ptr, which stores the starting points of the rows of the
   *arrays value and col_ind (size M+1)
   * @param colind col_ind, which stores the column numbers of the non-zero
   *elements (size nnz)
   * @param value value index, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: (M+1)+2nnz + (M+1)+nnz (compute hash)
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
      const int *colind, const Float *value);

  /**
   * @brief Create CRS matrix from array, also compute the hash
   * @param M # of row
   * @param N # of col
   * @param NNZ # of non-zero elements
   * @param rowptr row_ptr, which stores the starting points of the rows of the
   *arrays value and col_ind (size M+1)
   * @param colind n-origin col_ind, which stores the column numbers of the
   *non-zero elements (size nnz)
   * @param value value index, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: (M+1)+2nnz + (M+1)+nnz (compute hash) + nnz(compute
   *origin)
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
      const int *colind, const Float *value, const size_t origin);

  /**
   * @brief Create CRS matrix from std::vector, also compute the hash
   * @param M # of row
   * @param N # of col
   * @param rowptr row_ptr, which stores the starting points of the rows of the
   *arrays value and col_ind (size M+1)
   * @param colind col_ind, which stores the column numbers of the non-zero
   *elements (size nnz)
   * @param value value index, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: (M+1)+2nnz + (M+1)+nnz (compute hash)
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  CRS(const size_t M, const size_t N, const std::vector<int> &rowptr,
      const std::vector<int> &colind, const std::vector<Float> &value);

  /**
   * @brief Create CRS matrix from std::vector, also compute the hash
   * @param M # of row
   * @param N # of col
   * @param rowptr row_ptr, which stores the starting points of the rows of the
   *arrays value and col_ind (size M+1)
   * @param colind col_ind, which stores the column numbers of the non-zero
   *elements (size nnz)
   * @param value value index, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: (M+1)+2nnz + (M+1)+nnz (compute hash)
   * - Multi-threading: false
   * - GPU acceleration: true
   **/
  CRS(const size_t M, const size_t N, const std::vector<int> &rowptr,
      const std::vector<int> &colind, const vector<Float> &value);

  /**
   * @brief Convert CRS matrix from COO matrix, also compute the hash
   * @param coo COO format matrix
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void convert(COO<Float> &coo);

  /**
   * @brief Convert CRS matrix from COO matrix
   * @param crs CRS format matrix
   * @note
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(CRS<Float> &crs);

  /**
   * @brief Create CRS matrix from COO matrix, also compute the hash
   * @param coo Source COO format matrix
   * @return coo COO format matrix
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  CRS(COO<Float> &coo) { convert(coo); }

  /**
   * @brief Create CRS matrix from CRS matrix
   * @param mat CRS format matrix
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer:  (M+1)+2nnz (only allocation)
   *        - if `mat.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  CRS(const CRS<Float> &mat);

  /**
   * @brief Set CRS array from std::vector
   * @param M # of row
   * @param N # of col
   * @param rowptr row_ptr, which stores the starting points of the rows of the
   *arrays value and col_ind (size M+1)
   * @param colind col_ind, which stores the column numbers of the non-zero
   *elements (size nnz)
   * @param value value index, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: 3
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const size_t M, const size_t N, const std::vector<int> &rowptr,
               const std::vector<int> &colind, const std::vector<Float> &value);

  /**
   * @brief print all elements to standard I/O
   * @param force_cpu Ignore device status and output CPU data
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void print_all(bool force_cpu = false) const;

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
  [[nodiscard]] size_t get_nnz() const { return nnz; }

  /**
   * @brief get format name "CRS"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const { return "CRS"; }

  /**
   * @brief compute index array hash (to compare structure)
   * @note
   * - # of computation: nnz + rowN + 1
   * - Multi-threading: true
   * - GPU acceleration: true
   */
  void compute_hash();

  /**
   * @brief get index array hash (to compare structure)
   * @note
   * - # of computation: 1
   */
  [[nodiscard]] size_t get_hash() const { return structure_hash; }

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
  [[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }

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

  /*
   * @brief Memory data space required by the matrix
   * @note
   * - # of computation: 3
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] double get_data_size() const {
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
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer:
   *        - if `gpu_status == true`; coping data on GPU
   *        - else; coping data on CPU
   **/
  void operator=(const CRS<Float> &mat);

  /**
   * @brief Comparing matricies (A == mat)
   * @param mat CRS matrix
   * @param compare_cpu_and_device compare data on both CPU and GPU
   * @return true or false
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  [[nodiscard]] bool equal(const CRS<Float> &mat,
                           bool compare_cpu_and_device = false) const;

  /**
   * @brief Comparing matricies (A == mat)
   * @param mat CRS matrix
   * @return true or false
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator==(const CRS<Float> &mat) const;

  /**
   * @brief Comparing matricies (A != mat)
   * @param mat CRS matrix
   * @return true or false
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator!=(const CRS<Float> &mat) const;
};
} // namespace matrix
} // namespace monolish
