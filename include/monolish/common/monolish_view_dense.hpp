#pragma once
#include "./monolish_logger.hpp"
#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
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

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {
template <typename Float> class vector;

namespace matrix {
template <typename Float> class Dense;
template <typename Float> class CRS;
template <typename Float> class LinearOperator;
} // namespace matrix

namespace tensor {
template <typename Float> class tensor_Dense;
} // namespace tensor

/**
 * @addtogroup View_Dense_class
 * @{
 */

/**
 * @brief Dense view class
 * @note
 * - Multi-threading: true
 * - GPU acceleration: true
 */
template <typename TYPE, typename Float> class view_Dense {
private:
  TYPE &target;
  Float *target_data;
  size_t first;
  size_t last;
  size_t range;

  size_t rowN;
  size_t colN;

public:
  /**
   * @brief create view_Dense from vector(start:start+range)
   * @param x vector
   * @param start start position
   * @param row # of row
   * @param col # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_Dense(vector<Float> &x, const size_t start, const size_t row,
             const size_t col)
      : target(x) {
    first = start;
    rowN = row;
    colN = col;
    range = rowN * colN;
    last = start + range;
    target_data = x.data();
  }

  /**
   * @brief create view_Dense from Dense matrix(start:start+range)
   * @param A Dense matrix
   * @param start start position
   * @param row # of row
   * @param col # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_Dense(matrix::Dense<Float> &A, const size_t start, const size_t row,
             const size_t col)
      : target(A) {
    first = start;
    rowN = row;
    colN = col;
    range = rowN * colN;
    last = start + range;
    target_data = A.data();
  }

  /**
   * @brief create view_Dense from Dense tensor(start:start+range)
   * @param A Dense matrix
   * @param start start position
   * @param row # of row
   * @param col # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_Dense(tensor::tensor_Dense<Float> &A, const size_t start,
             const size_t row, const size_t col)
      : target(A) {
    first = start;
    rowN = row;
    colN = col;
    range = rowN * colN;
    last = start + range;
    target_data = A.data();
  }

  /**
   * @brief create view_Dense from monolish::vector(start:start+range)
   * @param x view_Dense create from monolish::vector
   * @param start start position (x.first + start)
   * @param row # of row
   * @param col # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_Dense(view_Dense<vector<Float>, Float> &x, const size_t start,
             const size_t row, const size_t col)
      : target(x) {
    first = x.get_first() + start;
    rowN = row;
    colN = col;
    range = rowN * colN;
    last = first + range;
    target_data = x.data();
  }

  /**
   * @brief create view_Dense from monolish::matrix::Dense(start:start+range)
   * @param x view_Dense create from monolish::matrix::Dense
   * @param start start position (x.first + start)
   * @param row # of row
   * @param col # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_Dense(view_Dense<matrix::Dense<Float>, Float> &x, const size_t start,
             const size_t row, const size_t col)
      : target(x) {
    first = x.get_first() + start;
    rowN = row;
    colN = col;
    range = rowN * colN;
    last = first + range;
    target_data = x.data();
  }

  /**
   * @brief create view_Dense from
   *monolish::tensor::tensor_Dense(start:start+range)
   * @param x view_Dense create from monolish::tensor::tensor_Dense
   * @param start start position (x.first + start)
   * @param row # of row
   * @param col # of col
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_Dense(view_Dense<tensor::tensor_Dense<Float>, Float> &x,
             const size_t start, const size_t row, const size_t col)
      : target(x) {
    first = x.get_first() + start;
    rowN = row;
    colN = col;
    range = rowN * colN;
    last = first + range;
    target_data = x.data();
  }

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
   * @brief get view_Dense size (range)
   * @return view_Dense size
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] size_t size() const { return range; }

  /**
   * @brief get view_Dense size (same as size())
   * @return view_Dense size
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] size_t get_nnz() const { return range; }

  /**
   * @brief get first position
   * @return first position
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] size_t get_first() const { return first; }

  /**
   * @brief get end position
   * @return end position
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] size_t get_last() const { return last; }

  /**
   * @brief get first position (same as get_first())
   * @return first position
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] size_t get_offset() const { return first; }

  /**
   * @brief change first position
   * @note
   * - # of computation: 1
   **/
  void set_first(size_t i) { first = i; }

  /**
   * @brief change last position
   * @note
   * - # of computation: 1
   **/
  void set_last(size_t i) {
    assert(first + i <= target.get_nnz());
    last = i;
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
  [[nodiscard]] Float at(const size_t i) const { return target.at(first + i); }

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
  [[nodiscard]] Float at(const size_t i, const size_t j) const {
    return target.at(first + i * get_col() + j);
  }

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
    return static_cast<const view_Dense<TYPE, Float> *>(this)->at(i);
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
    return static_cast<const view_Dense<TYPE, Float> *>(this)->at(i, j);
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
  void insert(const size_t i, const Float Val) {
    target.insert(first + i, Val);
    return;
  };

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
  void insert(const size_t i, const size_t j, const Float Val) {
    target.insert(first + i * get_col() + j, Val);
    return;
  };

  /**
   * @brief true: sended, false: not send
   * @return gpu status
   * @note
   * - # of data transfer: 0
   * - Multi-threading: false
   * - GPU acceleration: true
   **/
  [[nodiscard]] size_t get_device_mem_stat() const {
    return target.get_device_mem_stat();
  }

  /**
   * @brief returns a direct pointer to the original vector (dont include
   *offset)
   * @return A pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *data() const { return target_data; }

  /**
   * @brief returns a direct pointer to the vector (dont include offset)
   * @return A pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *data() { return target_data; }

  /**
   * @brief returns begin iterator (include offset)
   * @return A pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *begin() const { return target_data + get_offset(); }

  /**
   * @brief returns begin iterator (include offset)
   * @return A pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *begin() { return target_data + get_offset(); }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *end() const { return target_data + range; }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *end() { return target_data + range; }

  /**
   * @brief print all elements to standart I/O
   * @param force_cpu Ignore device status and output CPU data
   * @note
   * - # of computation: end-start
   * - Multi-threading: false
   * - GPU acceleration: true (doesn't work in parallel)
   **/
  void print_all(bool force_cpu = false) const;

  /**
   * @brief change last postion
   * @param N vector length
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: true
   * @warning
   * Cannot be resized beyond the range of the referenced vector
   **/
  void resize(size_t row, size_t col) {
    assert(first + row * col <= target.get_nnz());
    rowN = row;
    colN = col;
    range = rowN * colN;
    last = first + range;
  }

  /**
   * @brief reference to the element at position
   * @param i Position of an element in the vector
   * @return vector element (v[i])
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float &operator[](const size_t i) {
    if (target.get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    return target_data[i + first];
  }

  /**
   * @brief fill vector elements with a scalar value
   * @param value scalar value
   * @note
   * - # of computation: N
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void fill(Float value);
};
/**@}*/

} // namespace monolish
