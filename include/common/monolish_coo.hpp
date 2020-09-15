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
 * @brief Coodinate format Matrix (need to sort)
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
  std::vector<int> row_index;
  std::vector<int> col_index;
  std::vector<Float> val;

  COO()
      : rowN(0), colN(0), nnz(0), gpu_status(false), row_index(), col_index(),
        val() {}

  COO(const size_t M, const size_t N)
      : rowN(M), colN(N), nnz(0), gpu_status(false), row_index(), col_index(),
        val() {}

  COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
      const int *col, const Float *value)
      : rowN(M), colN(N), nnz(NNZ), gpu_status(false), row_index(nnz),
        col_index(nnz), val(nnz) {
    std::copy(row, row + nnz, row_index.begin());
    std::copy(col, col + nnz, col_index.begin());
    std::copy(value, value + nnz, val.begin());
  }

  // for n-origin
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

  COO(const matrix::COO<Float> &coo)
      : rowN(coo.get_row()), colN(coo.get_col()), nnz(coo.get_nnz()),
        gpu_status(false), row_index(nnz), col_index(nnz), val(nnz) {
    std::copy(coo.row_index.data(), coo.row_index.data() + nnz,
              row_index.begin());
    std::copy(coo.col_index.data(), coo.col_index.data() + nnz,
              col_index.begin());
    std::copy(coo.val.data(), coo.val.data() + nnz, val.begin());
  }

  void convert(const matrix::CRS<Float> &crs);
  COO(const matrix::CRS<Float> &crs) { convert(crs); }

  void convert(const matrix::Dense<Float> &dense);
  COO(const matrix::Dense<Float> &dense) { convert(dense); }

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   **/
  void send() const {
    throw std::runtime_error("error, GPU util of COO format is not impl. ");
  };

  /**
   * @brief recv data from GPU
   **/
  void recv() const {
    throw std::runtime_error("error, GPU util of COO format is not impl. ");
  };

  /**
   * @brief free data on GPU
   **/
  void device_free() const {};

  /**
   * @brief false; // true: sended, false: not send
   * @return true is sended.
   * **/
  bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief; free gpu mem.
   * **/
  ~COO() {
    if (get_device_mem_stat()) {
      device_free();
    }
  }

  // I/O
  // ///////////////////////////////////////////////////////////////////////////

  void set_row(const size_t M) { rowN = M; };
  void set_col(const size_t N) { colN = N; };
  void set_nnz(const size_t NNZ) { nnz = NNZ; };

  void input_mm(const char *filename);

  COO(const char *filename) { input_mm(filename); }

  /**
   * @brief print all elements to standart I/O
   **/
  void print_all() const;

  /**
   * @brief print all elements to file
   * @param[in] filename output filename
   **/
  void print_all(std::string filename) const;

  Float at(size_t i, size_t j);
  Float at(size_t i, size_t j) const;

  void set_ptr(size_t rN, size_t cN, std::vector<int> &r, std::vector<int> &c,
               std::vector<Float> &v);

  // not logging, only square
  // size_t size() const {return get_row() > get_col() ? get_row() : get_col();}
  size_t get_row() const { return rowN; }
  size_t get_col() const { return colN; }
  size_t get_nnz() const { return nnz; }
  std::string get_format_name() const { return "COO"; }

  /**
   * @brief matrix copy
   * @return copied COO matrix
   **/
  COO copy() {
    COO tmp(rowN, colN, nnz, row_index.data(), col_index.data(), val.data());
    return tmp;
  }

  std::vector<int> &get_row_ptr() { return row_index; }
  std::vector<int> &get_col_ind() { return col_index; }
  std::vector<Float> &get_val_ptr() { return val; }

  const std::vector<int> &get_row_ptr() const { return row_index; }
  const std::vector<int> &get_col_ind() const { return col_index; }
  const std::vector<Float> &get_val_ptr() const { return val; }

  // Utility
  // ///////////////////////////////////////////////////////////////////////////

  COO &transpose() {
    using std::swap;
    swap(rowN, colN);
    swap(row_index, col_index);
    return *this;
  }

  void transpose(COO &B) const {
    B.set_row(get_col());
    B.set_col(get_row());
    B.set_nnz(get_nnz());
    B.row_index = get_col_ind();
    B.col_index = get_row_ptr();
    B.val = get_val_ptr();
  }

  /**
   * @brief get data size [GB]
   * @return data size
   **/
  double get_data_size() const {
    return 3 * get_nnz() * sizeof(Float) / 1.0e+9;
  }

  std::string type() const { return "COO"; }

  std::vector<Float> row(std::size_t i) const;
  void row(std::size_t i, vector<Float> &vec) const;

  std::vector<Float> col(std::size_t j) const;
  void col(std::size_t i, vector<Float> &vec) const;

  std::vector<Float> diag() const;
  void diag(vector<Float> &vec) const;

  /////////////////////////////////////////////////////////////////////////////

  /**
   * @brief copy matrix, It is same as copy()
   * @param[in] filename source
   * @return output vector
   **/
  void operator=(const COO<Float> &mat) { mat = copy(); }

  /**
   * @brief insert element to (m, n)
   * @param[in] size_t m row number
   * @param[in] size_t n col number
   * @param[in] Float val matrix value (if multiple element exists, value will
   *be added together)
   **/
  void insert(size_t m, size_t n, Float val);

private:
  void _q_sort(int lo, int hi);

public:
  /**
   * @brief sort COO matrix elements (and merge elements)
   **/
  void sort(bool merge);
};

} // namespace matrix
} // namespace monolish
