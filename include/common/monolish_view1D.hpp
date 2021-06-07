/**
 * @autor RICOS Co. Ltd.
 * @file monolish_view1D.h
 * @brief declare view 1D class
 * @date 2019
 **/

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

/**
 * @brief 1D view class
 * @note
 * - Multi-threading: true
 * - GPU acceleration: true
 */
template <typename TYPE, typename Float> class view1D {
private:
  TYPE &target;
  Float *target_data;
  size_t first;
  size_t last;
  size_t range;

public:
  /**
   * @brief create view1D(start:end) from vector
   * @param x vector
   * @param start start position
   * @param end end position
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view1D(vector<Float> &x, const size_t start, const size_t end) : target(x) {
    first = start;
    last = end;
    range = last - first;
    target_data = x.data();
  }

  /**
   * @brief create view1D(start:end) from Dense matrix
   * @param A Dense matrix
   * @param start start position
   * @param end end position
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view1D(matrix::Dense<Float> &A, const size_t start, const size_t end)
      : target(A) {
    first = start;
    last = end;
    range = last - first;
    target_data = A.val.data();
  }

  /**
   * @brief create view1D(start:end) from monolish::vector
   * @param x view1D create from monolish::vector
   * @param start start position (x.first + start)
   * @param end end position (x.last + end)
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view1D(view1D<vector<Float>, Float> &x, const size_t start, const size_t end)
      : target(x) {
    first = x.get_first() + start;
    last = x.get_last() + end;
    range = last - first;
    target_data = x.data();
  }

  /**
   * @brief create view1D(start:end) from monolish::matrix::Dense
   * @param x view1D create from monolish::matrix::Dense
   * @param start start position (x.first + start)
   * @param end end position (x.last + end)
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view1D(view1D<matrix::Dense<Float>, Float> &x, const size_t start,
         const size_t end)
      : target(x) {
    first = x.get_first() + start;
    last = x.get_last() + end;
    range = last - first;
    target_data = x.data();
  }

  /**
   * @brief get view1D size (end-start)
   * @return view1D size
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] size_t size() const { return range; }

  /**
   * @brief get view1D size (same as size())
   * @return view1D size
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
  void resize(size_t N) {
    assert(first + N <= target.get_nnz());
    last = first + N;
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

} // namespace monolish
