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

#define MM_BANNER "%%MatrixMarket"
#define MM_MAT "matrix"
#define MM_VEC "vector"
#define MM_FMT "coordinate"
#define MM_TYPE_REAL "real"
#define MM_TYPE_GENERAL "general"
#define MM_TYPE_SYMM "symmetric"

namespace monolish {
template <typename Float> class vector;
namespace matrix {
template <typename Float> class Dense;
template <typename Float> class CRS;

/**
 * @brief Coodinate (COO) format Matrix (need to sort)
 * @note
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): false
 */
template <typename Float> class COO {
private:
  /**
   * @brief neet col = row now
   */
  size_t rowN;
  size_t colN;
  size_t nnz;

  mutable bool gpu_status = false; // true: sended, false: not send

public:
  /**
   * @brief Coodinate format row index, which stores row numbers of the non-zero
   * elements (size nnz)
   */
  std::vector<int> row_index;

  /**
   * @brief Coodinate format column index, which stores column numbers of the
   * non-zero elements (size nnz)
   */
  std::vector<int> col_index;

  /**
   * @brief Coodinate format value array, which stores values of the non-zero
   * elements (size nnz)
   */
  std::vector<Float> val;

  COO()
      : rowN(0), colN(0), nnz(0), gpu_status(false), row_index(), col_index(),
        val() {}

  /**
   * @brief Initialize M x N COO matrix
   * @param M # of row
   * @param N # of col
   * @note
   * - # of computation: 0
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  COO(const size_t M, const size_t N)
      : rowN(M), colN(N), nnz(0), gpu_status(false), row_index(), col_index(),
        val() {}

  /**
   * @brief Create COO matrix from array
   * @param M # of row
   * @param N # of col
   * @param NNZ # of non-zero elements
   * @param row row index, which stores the row numbers of the non-zero elements
   *(size nnz)
   * @param col col index, which stores the column numbers of the non-zero
   *elements (size nnz)
   * @param value value index, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
      const int *col, const Float *value)
      : rowN(M), colN(N), nnz(NNZ), gpu_status(false), row_index(nnz),
        col_index(nnz), val(nnz) {
    std::copy(row, row + nnz, row_index.begin());
    std::copy(col, col + nnz, col_index.begin());
    std::copy(value, value + nnz, val.begin());
  }

  /**
   * @brief Create COO matrix from n-origin array
   * @param M # of row
   * @param N # of col
   * @param NNZ # of non-zero elements
   * @param row n-origin row index, which stores the row numbers of the non-zero
   *elements (size nnz)
   * @param col n-origin col index, which stores the column numbers of the
   *non-zero elements (size nnz)
   * @param value n-origin value index, which stores the non-zero elements (size
   *nnz)
   * @param origin n-origin
   * @note
   * - # of computation: 3nnz + 2nnz(adjust possition using origin)
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): false
   **/
  COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
      const int *col, const Float *value, const size_t origin)
      : rowN(M), colN(N), nnz(NNZ), gpu_status(false), row_index(nnz),
        col_index(nnz), val(nnz) {
    std::copy(row, row + nnz, row_index.begin());
    std::copy(col, col + nnz, col_index.begin());
    std::copy(value, value + nnz, val.begin());

#pragma omp parallel for
    for (size_t i = 0; i < nnz; i++) {
      row_index[i] -= origin;
      col_index[i] -= origin;
    }
  }

  /**
   * @brief Create COO matrix from COO matrix
   * @param coo input COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  COO(const matrix::COO<Float> &coo)
      : rowN(coo.get_row()), colN(coo.get_col()), nnz(coo.get_nnz()),
        gpu_status(false), row_index(nnz), col_index(nnz), val(nnz) {
    std::copy(coo.row_index.data(), coo.row_index.data() + nnz,
              row_index.begin());
    std::copy(coo.col_index.data(), coo.col_index.data() + nnz,
              col_index.begin());
    std::copy(coo.val.data(), coo.val.data() + nnz, val.begin());
  }

  /**
   * @brief Create COO matrix from CRS matrix
   * @param crs input COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void convert(const matrix::CRS<Float> &crs);

  /**
   * @brief Create COO matrix from CRS matrix
   * @param crs input COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  COO(const matrix::CRS<Float> &crs) { convert(crs); }

  /**
   * @brief Create COO matrix from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * @note
   * - # of computation: 3NM
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void convert(const matrix::Dense<Float> &dense);

  /**
   * @brief Create COO matrix from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * @note
   * - # of computation: 3NM
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  COO(const matrix::Dense<Float> &dense) { convert(dense); }

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @warning
   * COO format can not use GPU function
   **/
  void send() const {
    throw std::runtime_error("error, GPU util of COO format is not impl. ");
  };

  /**
   * @brief recv data from GPU
   * @warning
   * COO format can not use GPU function
   **/
  void recv() const {
    throw std::runtime_error("error, GPU util of COO format is not impl. ");
  };

  /**
   * @brief free data on GPU
   * @warning
   * COO format can not use GPU function
   **/
  void device_free() const {};

  /**
   * @brief false; // true: sended, false: not send
   * @return true is sended.
   * @warning
   * COO format can not use GPU function
   * **/
  bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief; free gpu mem.
   * @warning
   * COO format can not use GPU function
   * **/
  ~COO() {
    if (get_device_mem_stat()) {
      device_free();
    }
  }

  // I/O
  // ///////////////////////////////////////////////////////////////////////////

  /**
   * @brief Set row number
   * @param M # of row
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void set_row(const size_t M) { rowN = M; };

  /**
   * @brief Set col number
   * @param N # of col
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void set_col(const size_t N) { colN = N; };

  /**
   * @brief Set # of non-zero elements
   * @param NNZ # of non-zero elements
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void set_nnz(const size_t NNZ) { nnz = NNZ; };

  /**
   * @brief Create COO matrix from MatrixMatrket format file
   * @param filename MatrixMarket format file name
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void input_mm(const char *filename);

  /**
   * @brief Create COO matrix from MatrixMatrket format file
   * @param filename MatrixMarket format file name
   * @note
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  COO(const char *filename) { input_mm(filename); }

  /**
   * @brief print all elements to standart I/O
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void print_all() const;

  /**
   * @brief print all elements to file
   * @param filename output filename
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void print_all(std::string filename) const;

  /**
   * @brief Get matrix element (A(i,j))
   * @note
   * - # of computation: i*M+j
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  Float at(size_t i, size_t j);

  /**
   * @brief Get matrix element (A(i,j))
   * @note
   * - # of computation: i*M+j
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  Float at(size_t i, size_t j) const;

  /**
   * @brief Set COO array from std::vector
   * @param rN # of row
   * @param cN # of column
   * @param r row_index
   * @param c col_index
   * @param v value
   * @note
   * - # of computation: 3
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void set_ptr(size_t rN, size_t cN, std::vector<int> &r, std::vector<int> &c,
               std::vector<Float> &v);

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
  size_t get_nnz() const { return nnz; }

  /**
   * @brief matrix copy
   * @return copied COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  COO copy() {
    COO tmp(rowN, colN, nnz, row_index.data(), col_index.data(), val.data());
    return tmp;
  }

  /**
   * @brief get row index
   * @return row index
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::vector<int> &get_row_ptr() { return row_index; }

  /**
   * @brief get column index
   * @return column index
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::vector<int> &get_col_ind() { return col_index; }

  /*
   * @brief get value
   * @return velue
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::vector<Float> &get_val_ptr() { return val; }

  /**
   * @brief get row index
   * @return row index
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  const std::vector<int> &get_row_ptr() const { return row_index; }

  /**
   * @brief get column index
   * @return column index
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  const std::vector<int> &get_col_ind() const { return col_index; }

  /*
   * @brief get value
   * @return velue
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  const std::vector<Float> &get_val_ptr() const { return val; }

  // Utility
  // ///////////////////////////////////////////////////////////////////////////

  /*
   * @brief get transposed matrix (A^T)
   * @return tranposed matrix A^T
   * @note
   * - # of computation: 2
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  COO &transpose() {
    using std::swap;
    swap(rowN, colN);
    swap(row_index, col_index);
    return *this;
  }

  /*
   * @brief create transposed matrix from COO matrix (A = B^T)
   * @param B COO matrix
   * @note
   * - # of computation: 3 * nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void transpose(COO &B) const {
    B.set_row(get_col());
    B.set_col(get_row());
    B.set_nnz(get_nnz());
    B.row_index = get_col_ind();
    B.col_index = get_row_ptr();
    B.val = get_val_ptr();
  }

  /*
   * @brief Memory data space required by the matrix
   * @note
   * - # of computation: 3
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  double get_data_size() const {
    return 3 * get_nnz() * sizeof(Float) / 1.0e+9;
  }

  /**
   * @brief get format name "COO"
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::string type() const { return "COO"; }

  /**
   * @brief get row vector
   * @param i row number
   * @return row vector
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::vector<Float> row(std::size_t i) const;

  /**
   * @brief get row vector
   * @param i row number
   * @param vec row vector
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void row(std::size_t i, vector<Float> &vec) const;

  /**
   * @brief get column vector
   * @param j column number
   * @return column vector
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::vector<Float> col(std::size_t j) const;

  /**
   * @brief get column vector
   * @param i column number
   * @param vec column vector
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void col(std::size_t i, vector<Float> &vec) const;

  /**
   * @brief get diag. vector
   * @return diag. vector
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  std::vector<Float> diag() const;

  /**
   * @brief get diag. vector
   * @param vec diag. vector
   * @note
   * - # of computation: nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void diag(vector<Float> &vec) const;

  /////////////////////////////////////////////////////////////////////////////

  /**
   * @brief matrix copy
   * @note
   * - # of computation: 3nnz
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void operator=(const COO<Float> &mat) { mat = copy(); }

  /**
   * @brief insert element to (m, n)
   * @param m row number
   * @param n col number
   * @param val matrix value (if multiple element exists, value will
   *be added together)
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   * @warning
   * This function does not check for duplicate values.
   * This adds an element to the end of the array.
   * In most cases, calling sort() is required after this function.
   **/
  void insert(size_t m, size_t n, Float val);

private:
  void _q_sort(int lo, int hi);

public:
  /**
   * @brief sort COO matrix elements (and merge elements)
   * @param merge neet to merge (true or false)
   * @note
   * - # of computation: 3nnz x log(3nnz) ~ 3nnz^2
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void sort(bool merge);
};

} // namespace matrix
} // namespace monolish
