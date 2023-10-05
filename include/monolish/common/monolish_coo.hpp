#pragma once
#include <exception>
#include <memory>
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

#define MM_BANNER "%%MatrixMarket"
#define MM_MAT "matrix"
#define MM_VEC "vector"
#define MM_FMT "coordinate"
#define MM_TYPE_REAL "real"
#define MM_TYPE_GENERAL "general"
#define MM_TYPE_SYMM "symmetric"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_Dense;
}
namespace matrix {
template <typename Float> class Dense;
template <typename Float> class CRS;
template <typename Float> class LinearOperator;

/**
 * @addtogroup COO_class
 * @{
 */

/**
 * @brief Coodinate (COO) format Matrix (need to sort)
 * @note
 * - Multi-threading: true
 * - GPU acceleration: false
 */
template <typename Float> class COO {
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
  // size_t nnz;

  /**
   * @brief true: sended, false: not send
   */
  mutable bool gpu_status = false;

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
   * @brief Coodinate format value array (pointer), which stores values of the
   * non-zero elements
   */
  std::shared_ptr<Float> val;

  /**
   * @brief # of non-zero element
   */
  size_t val_nnz = 0;

  /**
   * @brief alloced matrix size
   */
  std::size_t alloc_nnz = 0;

  /**
   * @brief matrix create flag;
   */
  bool val_create_flag = false;

  COO()
      : rowN(0), colN(0), gpu_status(false), row_index(), col_index(),
        val_nnz(0) {
    val_create_flag = true;
  }

  /**
   * @brief Initialize M x N COO matrix
   * @param M # of row
   * @param N # of col
   * @note
   * - # of computation: 0
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  COO(const size_t M, const size_t N)
      : rowN(M), colN(N), gpu_status(false), row_index(), col_index(),
        val_nnz(0) {
    val_create_flag = true;
  }

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
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
      const int *col, const Float *value);

  /**
   * @brief Create COO matrix from std::vector
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
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  COO(const size_t M, const size_t N, const size_t NNZ,
      const std::vector<int> &row, const std::vector<int> &col,
      const std::vector<Float> &value) {
    this = COO(M, N, NNZ, row.data(), col.data(), value.data());
  }

  /**
   * @brief Create COO matrix from monolish::vector
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
   * - Multi-threading: false
   * - GPU acceleration: false
   * @warning
   * gpu_status of input vectors must be false
   **/
  COO(const size_t M, const size_t N, const size_t NNZ,
      const std::vector<int> &row, const std::vector<int> &col,
      const vector<Float> &value) {
    assert(value.get_device_mem_stat() == false);
    this = COO(M, N, NNZ, row.data(), col.data(), value.data());
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
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
      const int *col, const Float *value, const size_t origin);

  /**
   * @brief Create COO matrix from n-origin array
   * @param M # of row
   * @param N # of col
   * @param NNZ # of non-zero elements
   * @param row n-origin row index, which stores the row numbers of the non-zero
   *elements (size nnz)
   * @param col n-origin col index, which stores the column numbers of the
   *non-zero elements (size nnz)
   * @param value n-origin value, which stores the non-zero elements (size
   *nnz)
   * @param origin n-origin
   * @note
   * - # of computation: 3nnz + 2nnz(adjust possition using origin)
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  COO(const size_t M, const size_t N, const size_t NNZ,
      const std::vector<int> &row, const std::vector<int> &col,
      const std::vector<Float> &value, const size_t origin) {
    this = COO(M, N, NNZ, row.data(), col.data(), value.data(), origin);
  }

  /**
   * @brief Create COO matrix from COO matrix
   * @param coo input COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  COO(const matrix::COO<Float> &coo);

  /**
   * @brief Initialize COO matrix of the same size as input matrix
   * @param coo input COO matrix
   * @param value the value to initialize elements
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  COO(const matrix::COO<Float> &coo, Float value);

  /**
   * @brief Create COO matrix from CRS matrix
   * @param crs input COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void convert(const matrix::CRS<Float> &crs);

  /**
   * @brief Create COO matrix from CRS matrix
   * @param crs input COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  COO(const matrix::CRS<Float> &crs) {
    val_create_flag = true;
    convert(crs);
  }

  /**
   * @brief Create COO matrix from Dense matrix (drop zero)
   * @param dense input Dense matrix (size M x N)
   * @note
   * - # of computation: 3NM
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void convert(const matrix::Dense<Float> &dense);

  /**
   * @brief Create COO matrix from Dense matrix (drop zero)
   * @param dense input Dense matrix (size M x N)
   * @note
   * - # of computation: 3NM
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  COO(const matrix::Dense<Float> &dense) {
    val_create_flag = true;
    convert(dense);
  }

  void convert(const matrix::LinearOperator<Float> &linearoperator);

  COO(const matrix::LinearOperator<Float> &linearoperator) {
    val_create_flag = true;
    convert(linearoperator);
  }

  /**
   * @brief Set row number
   * @param M # of row
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_row(const size_t M) { rowN = M; };

  /**
   * @brief Set col number
   * @param N # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_col(const size_t N) { colN = N; };

  /**
   * @brief Set # of non-zero elements
   * @param NNZ # of non-zero elements
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  // void set_nnz(const size_t NNZ) { val_nnz = NNZ; };

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
  [[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief; free gpu mem.
   * @warning
   * COO format can not use GPU function
   * **/
  ~COO() {
    if (val_create_flag) {
      if (get_device_mem_stat()) {
        device_free();
      }
    }
  }

  /**
   * @brief returns a direct pointer to the vector
   * @return A const pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *data() const { return val.get(); }

  /**
   * @brief returns a direct pointer to the vector
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

      row_index.resize(N);
      col_index.resize(N);
    } else {
      throw std::runtime_error("Error, not create vector cant use resize");
    }
  }

  // I/O
  // ///////////////////////////////////////////////////////////////////////////

  /**
   * @brief Create COO matrix from MatrixMarket format file (only real general)
   * (MatrixMarket format: https://math.nist.gov/MatrixMarket/formats.html)
   * @param filename MatrixMarket format file name
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void input_mm(const std::string filename);

  /**
   * @brief Create COO matrix from MatrixMarket format file (only real general)
   * (MatrixMarket format: https://math.nist.gov/MatrixMarket/formats.html)
   * @param filename MatrixMarket format file name
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  COO(const std::string filename) { input_mm(filename); }

  /**
   * @brief output matrix elements in MatrixMarket format
   * (MatrixMarket format: https://math.nist.gov/MatrixMarket/formats.html)
   * @param filename MatrixMarket format file name
   * @note
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void output_mm(const std::string filename) const;

  /**
   * @brief print all elements to standard I/O
   * @param force_cpu Unused options for integrity
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void print_all(bool force_cpu = false) const;

  /**
   * @brief print all elements to file
   * @param filename output filename
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void print_all(const std::string filename) const;

  /**
   * @brief Get matrix element (A(i,j))
   * @note
   * - # of computation: i*M+j
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t i, const size_t j) const;

  /**
   * @brief Get matrix element (A(i,j))
   * @note
   * - # of computation: i*M+j
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t i, const size_t j) {
    return static_cast<const COO *>(this)->at(i, j);
  };

  /**
   * @brief Set COO array from std::vector
   * @param rN # of row
   * @param cN # of column
   * @param r row_index
   * @param c col_index
   * @param v value
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const size_t rN, const size_t cN, const std::vector<int> &r,
               const std::vector<int> &c, const std::vector<Float> &v);

  /**
   * @brief Set COO array from std::vector
   * @param rN # of row
   * @param cN # of column
   * @param r row_index
   * @param c col_index
   * @param v value
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const size_t rN, const size_t cN, const std::vector<int> &r,
               const std::vector<int> &c, const size_t vsize, const Float *v);

  /**
   * @brief Set COO array from std::vector
   * @param rN # of row
   * @param cN # of column
   * @param r row_index
   * @param c col_index
   * @param v value
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const size_t rN, const size_t cN, const std::vector<int> &r,
               const std::vector<int> &c, const size_t vsize, const Float v);

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
  [[nodiscard]] size_t get_nnz() const { return val_nnz; }

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
   * @brief get row index
   * @return row index
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::vector<int> &get_row_ptr() { return row_index; }

  /**
   * @brief get column index
   * @return column index
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::vector<int> &get_col_ind() { return col_index; }

  /**
   * @brief get value
   * @return value
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::vector<Float> get_val_ptr() {
    std::vector<Float> val(val_nnz);
    for (size_t i = 0; i < val_nnz; ++i) {
      val[i] = data()[i];
    }
    return val;
  }

  /**
   * @brief get row index
   * @return row index
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] const std::vector<int> &get_row_ptr() const {
    return row_index;
  }

  /**
   * @brief get column index
   * @return column index
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] const std::vector<int> &get_col_ind() const {
    return col_index;
  }

  /**
   * @brief get value
   * @return value
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] const std::vector<Float> get_val_ptr() const {
    std::vector<Float> val(val_nnz);
    for (size_t i = 0; i < val_nnz; ++i) {
      val[i] = data()[i];
    }
    return val;
  }

  // Utility
  // ///////////////////////////////////////////////////////////////////////////

  /**
   * @brief get transposed matrix (A^T)
   * @return tranposed matrix A^T
   * @note
   * - # of computation: 2 * nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void transpose();

  /**
   * @brief create transposed matrix from COO matrix (B = A^T)
   * @param B COO matrix
   * @note
   * - # of computation: 3 * nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void transpose(const COO &B);

  /**
   * @brief Memory data space required by the matrix
   * @note
   * - # of computation: 3
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  double get_data_size() const {
    return 3 * get_nnz() * sizeof(Float) / 1.0e+9;
  }

  /**
   * @brief get format name "COO"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const { return "COO"; }

  /**
   * @brief returns a begin iterator
   * @return begin iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *begin() const { return data(); }

  /**
   * @brief returns a begin iterator
   * @return begin iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *begin() { return data(); }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *end() const { return data() + get_nnz(); }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *end() { return data() + get_nnz(); }

  /**
   * @brief get diag. vector
   * @param vec diag. vector
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
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
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
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
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void col(const size_t c, vector<Float> &vec) const;
  void col(const size_t c, view1D<vector<Float>, Float> &vec) const;
  void col(const size_t c, view1D<matrix::Dense<Float>, Float> &vec) const;
  void col(const size_t c,
           view1D<tensor::tensor_Dense<Float>, Float> &vec) const;

  /////////////////////////////////////////////////////////////////////////////

  /**
   * @brief matrix copy
   * @param mat COO matrix
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   * @warning
   * src. and dst. must be same non-zero structure (dont check in this function)
   **/
  void operator=(const COO<Float> &mat);

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
    return data()[i];
  }

  /**
   * @brief Comparing matrices (A == mat)
   * @param mat COO matrix
   * @param compare_cpu_and_device Unused options for integrity
   * @return true or false
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  [[nodiscard]] bool equal(const COO<Float> &mat,
                           bool compare_cpu_and_device = false) const;

  /**
   * @brief Comparing matrices (A == mat)
   * @param mat COO matrix
   * @return true or false
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  [[nodiscard]] bool operator==(const COO<Float> &mat) const;

  /**
   * @brief Comparing matrices (A != mat)
   * @param mat COO matrix
   * @return true or false
   * @note
   * - # of computation: 3nnz
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  [[nodiscard]] bool operator!=(const COO<Float> &mat) const;

  /**
   * @brief insert element to (m, n)
   * @param m row number
   * @param n col number
   * @param val matrix value (if multiple element exists, value will
   *be added together)
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   * @warning
   * This function does not check for duplicate values.
   * This adds an element to the end of the array.
   * In most cases, calling sort() is required after this function.
   **/
  void insert(const size_t m, const size_t n, const Float val);

private:
  void _q_sort(int lo, int hi);

public:
  /**
   * @brief sort COO matrix elements (and merge elements)
   * @param merge need to merge (true or false)
   * @note
   * - # of computation: 3nnz x log(3nnz) ~ 3nnz^2
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void sort(bool merge);
};
/**@}*/

} // namespace matrix
} // namespace monolish
