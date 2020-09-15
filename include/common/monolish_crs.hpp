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
 * @brief CRS format Matrix
 */
template <typename Float> class CRS {
private:
  /**
   * @brief neet col = row now
   */
  size_t rowN;
  size_t colN;
  size_t nnz;

  mutable bool gpu_status = false; // true: sended, false: not send

public:
  std::vector<Float> val;
  std::vector<int> col_ind;
  std::vector<int> row_ptr;

  CRS() {}

  CRS(const size_t M, const size_t N, const size_t NNZ, const int *row,
      const int *col, const Float *value)
      : rowN(M), colN(N), nnz(NNZ), gpu_status(false), row_ptr(M + 1),
        col_ind(nnz), val(nnz) {
    std::copy(row, row + (M + 1), row_ptr.begin());
    std::copy(col, col + nnz, col_ind.begin());
    std::copy(value, value + nnz, val.begin());
  }

  void convert(COO<Float> &coo);
  CRS(COO<Float> &coo) { convert(coo); }

  void print_all();

  // size_t size() const {return rowN > colN ? rowN : colN;}
  size_t get_row() const { return rowN; }
  size_t get_col() const { return colN; }
  size_t get_nnz() const { return nnz; }

  std::string type() const { return "CRS"; }

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   **/
  void send() const;

  /**
   * @brief recv and free data from GPU
   **/
  void recv();

  /**
   * @brief recv data from GPU (w/o free)
   **/
  void nonfree_recv();

  /**
   * @brief free data on GPU
   **/
  void device_free() const;

  /**
   * @brief false; // true: sended, false: not send
   * @return true is sended.
   * **/
  bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief; free gpu mem.
   * **/
  ~CRS() {
    if (get_device_mem_stat()) {
      device_free();
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void diag(vector<Float> &vec) const;
  void row(const size_t r, vector<Float> &vec) const;
  void col(const size_t c, vector<Float> &vec) const;

  /////////////////////////////////////////////////////////////////////////////

  /**
   * @brief get data size [GB]
   * @return data size
   **/
  double get_data_size() const {
    return (get_nnz() * sizeof(Float) + (get_row() + 1) * sizeof(int) +
            get_nnz() * sizeof(int)) /
           1.0e+9;
  }

  /**
   * @brief matrix copy
   * @return copied CRS matrix
   **/
  CRS copy();

  CRS(const CRS<Float> &mat);

  /**
   * @brief copy matrix, It is same as copy()
   * @return output matrix
   **/
  void operator=(const CRS<Float> &mat);

  // mat - vec
  vector<Float> operator*(vector<Float> &vec);

  // mat - scalar
  CRS<Float> operator*(const Float value);

  // crs-dense
  Dense<Float> operator*(const Dense<Float> &B);

  // crs-dense
  CRS<Float> operator+(const CRS<Float> &B);
};
} // namespace matrix
} // namespace monolish
