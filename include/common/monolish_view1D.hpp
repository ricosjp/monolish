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
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <memory>

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
  view1D(monolish::vector<Float> &x, const size_t start, const size_t end)
      : target(x) {
    first = start;
    last = end;
    range = last - first;
    target_data = x.data();
  }

  view1D(monolish::matrix::Dense<Float> &A, const size_t start,
         const size_t end)
      : target(A) {
    first = start;
    last = end;
    range = last - first;
    target_data = A.val.data();
  }

  size_t size() const { return range; }
  size_t get_nnz() const { return range; }

  void set_first(size_t i) { first = i; }

  void set_last(size_t i) {
    assert(first + i <= target.get_nnz());
    last = i;
  }

  size_t get_device_mem_stat() const { return target.get_device_mem_stat(); }

  Float *data() { return target_data; }
  Float *data() const { return data(); }

  void print_all(bool force_cpu = false) const;

  void resize(size_t N) {
    assert(first + N <= target.get_nnz());
    last = first + N;
  }

  Float &operator[](const size_t i) {
    if (target.get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    return target_data[i + first];
  }
};

} // namespace monolish
