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
template <typename TYPE, typename Float> class view_tensor_Dense {
private:
  TYPE &target;
  Float *target_data;
  size_t first;
  size_t last;
  size_t range;

  std::vector<size_t> shape;

public:
  /**
   * @brief create view_tensor_Dense from vector(start:start+range)
   * @param x vector
   * @param start start position
   * @param shape_ shape of tensor
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_tensor_Dense(vector<Float> &x, const size_t start,
                    const std::vector<size_t> &shape_)
      : target(x) {
    first = start;
    shape = shape_;
    range = calc_range();
    last = start + range;
    target_data = x.data();
  }

  /**
   * @brief create view_tensor_Dense from Dense matrix(start:start+range)
   * @param A Dense matrix
   * @param start start position
   * @param shape_ shape of tensor
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_tensor_Dense(matrix::Dense<Float> &A, const size_t start,
                    const std::vector<size_t> &shape_)
      : target(A) {
    first = start;
    shape = shape_;
    range = calc_range();
    last = start + range;
    target_data = A.data();
  }

  /**
   * @brief create view_tensor_Dense from Dense tensor(start:start+range)
   * @param A Dense matrix
   * @param start start position
   * @param shape_ shape of tensor
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_tensor_Dense(tensor::tensor_Dense<Float> &A, const size_t start,
                    const std::vector<size_t> &shape_)
      : target(A) {
    first = start;
    shape = shape_;
    range = calc_range();
    last = start + range;
    target_data = A.data();
  }

  /**
   * @brief create view_tensor_Dense from monolish::vector(start:start+range)
   * @param x view_tensor_Dense create from monolish::vector
   * @param start start position (x.first + start)
   * @param shape_ shape of tensor
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_tensor_Dense(view_tensor_Dense<vector<Float>, Float> &x,
                    const size_t start, const std::vector<size_t> &shape_)
      : target(x) {
    first = x.get_first() + start;
    shape = shape_;
    range = calc_range();
    last = first + range;
    target_data = x.data();
  }

  /**
   * @brief create view_tensor_Dense from
   *monolish::matrix::Dense(start:start+range)
   * @param x view_tensor_Dense create from monolish::matrix::Dense
   * @param start start position (x.first + start)
   * @param shape_ shape of tensor
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_tensor_Dense(view_tensor_Dense<matrix::Dense<Float>, Float> &x,
                    const size_t start, const std::vector<size_t> &shape_)
      : target(x) {
    first = x.get_first() + start;
    shape = shape_;
    range = calc_range();
    last = first + range;
    target_data = x.data();
  }

  /**
   * @brief create view_tensor_Dense from
   *monolish::tensor::tensor_Dense(start:start+range)
   * @param x view_tensor_Dense create from monolish::tensor::tensor_Dense
   * @param start start position (x.first + start)
   * @param shape_ shape of tensor
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  view_tensor_Dense(view_tensor_Dense<tensor::tensor_Dense<Float>, Float> &x,
                    const size_t start, const std::vector<size_t> &shape_)
      : target(x) {
    first = x.get_first() + start;
    shape = shape_;
    range = calc_range();
    last = first + range;
    target_data = x.data();
  }

  /**
   * @brief get format name "view_tensor_Dense"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const {
    return "view_tensor_Dense(" + target.type() + ")";
  }

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @note
   * - # of data transfer: N
   * - Multi-threading: false
   * - GPU acceleration: true
   **/
  void send() const { target.send(); };

  /**
   * @brief recv data from GPU, and free data on GPU
   * @note
   * - # of data transfer: N
   * - Multi-threading: false
   * - GPU acceleration: true
   **/
  void recv() { target.recv(); };

  /**
   * @brief calculate view_tensor_Dense size from shape
   * @return view_tensor_Dense size
   * @note
   * - # of computation: dim
   */
  [[nodiscard]] size_t calc_range() const {
    size_t N = 1;
    for (auto n : shape) {
      N *= n;
    }
    return N;
  };

  /**
   * @brief get shape
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::vector<size_t> get_shape() const { return shape; }

  /**
   * @brief get shared_ptr of val
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::shared_ptr<Float> get_val() { return target.get_val(); }

  /**
   * @brief get view_tensor_Dense size (range)
   * @return view_tensor_Dense size
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] size_t size() const { return range; }

  /**
   * @brief get view_tensor_Dense size (same as size())
   * @return view_tensor_Dense size
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

  [[nodiscard]] std::shared_ptr<Float> get_val() const { return target.val; }

  [[nodiscard]] size_t get_alloc_nnz() const { return target.get_alloc_nnz(); }

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
   * @brief get element A[index]
   * @param pos aligned position index
   * @return A[index]
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t pos) const {
    return target.at(first + pos);
  }

  /**
   * @brief get element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const std::vector<size_t> &pos) const {
    return target.at(first + get_index(pos));
  }

  /**
   * @brief get element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  template <typename... Args>
  [[nodiscard]] Float at(const std::vector<size_t> &pos, const size_t dim,
                         const Args... args) const {
    std::vector<size_t> pos_copy = pos;
    pos_copy.push_back(dim);
    return this->at(pos_copy, args...);
  };

  /**
   * @brief get element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  template <typename... Args>
  [[nodiscard]] Float at(const size_t dim, const Args... args) const {
    std::vector<size_t> pos(1);
    pos[0] = dim;
    return this->at(pos, args...);
  };

  /**
   * @brief get element A[index] (only CPU)
   * @param pos aligned position index
   * @return A[index]
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t pos) {
    return static_cast<const view_tensor_Dense<TYPE, Float> *>(this)->at(pos);
  };

  /**
   * @brief get element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const std::vector<size_t> &pos) {
    return static_cast<const view_tensor_Dense<TYPE, Float> *>(this)->at(pos);
  };

  /**
   * @brief get element A[pos[0]][pos[1]]... (onlu CPU)
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  template <typename... Args>
  [[nodiscard]] Float at(const std::vector<size_t> &pos, const Args... args) {
    return static_cast<const view_tensor_Dense *>(this)->at(pos, args...);
  };

  /**
   * @brief get element A[pos[0]][pos[1]]... (onlu CPU)
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  template <typename... Args>
  [[nodiscard]] Float at(const size_t dim, const Args... args) {
    return static_cast<const view_tensor_Dense *>(this)->at(dim, args...);
  };

  /**
   * @brief set element A[index]...
   * @param pos aligned position index
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
   * @brief set element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @param Val scalar value
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void insert(const std::vector<size_t> &pos, const Float Val) {
    target.insert(first + get_index(pos), Val);
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
   * @brief gpu status shared pointer
   * @return gpu status shared pointer
   */
  [[nodiscard]] std::shared_ptr<bool> get_gpu_status() const {
    return target.get_gpu_status();
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
   * @brief returns a reference of the target
   * @return target
   * @note
   * - # of computation: 1
   */
  [[nodiscard]] TYPE &get_target() const { return target; }

  /**
   * @brief returns a reference of the target
   * @return target
   * @note
   * - # of computation: 1
   */
  [[nodiscard]] TYPE &get_target() { return target; }

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
  void resize(std::vector<size_t> &shape_) {
    shape = shape_;
    range = calc_range();
    assert(first + range <= target.get_nnz());
    last = first + range;
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

  /**
   * @brief tensor copy
   * @param tens Dense tensor
   * @return copied dense tensor
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: 0
   *        - if `gpu_statius == true`; coping data on CPU
   *        - else; coping data on CPU
   **/
  void operator=(const tensor::tensor_Dense<Float> &tens);

  /**
   * @brief tensor copy
   * @param tens Dense tensor
   * @return copied dense tensor
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: 0
   *        - if `gpu_statius == true`; coping data on CPU
   *        - else; coping data on CPU
   **/
  void operator=(const view_tensor_Dense<vector<Float>, Float> &tens);

  /**
   * @brief tensor copy
   * @param tens Dense tensor
   * @return copied dense tensor
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: 0
   *        - if `gpu_statius == true`; coping data on CPU
   *        - else; coping data on CPU
   **/
  void operator=(const view_tensor_Dense<matrix::Dense<Float>, Float> &tens);

  /**
   * @brief tensor copy
   * @param tens Dense tensor
   * @return copied dense tensor
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: 0
   *        - if `gpu_statius == true`; coping data on CPU
   *        - else; coping data on CPU
   **/
  void
  operator=(const view_tensor_Dense<tensor::tensor_Dense<Float>, Float> &tens);

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
   * @brief get aligned index from vector index (A[pos] = A[ind[0]][ind[1]]...)
   * @param pos position (std::vector)
   * @return aligned position
   * @note
   * - # of computation: shape size
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  size_t get_index(const std::vector<size_t> &pos) const {
    if (pos.size() != this->shape.size()) {
      throw std::runtime_error("pos size should be same with the shape");
    }
    size_t ind = 0;
    for (auto i = 0; i < pos.size(); ++i) {
      ind *= this->shape[i];
      ind += pos[i];
    }
    return ind;
  }

  /**
   * @brief get vector index from aligned index (A[pos[0]][pos[1]]... = A[ind])
   * @param pos position (scalar)
   * @return vector position
   * @note
   * - # of computation: shape size
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  std::vector<size_t> get_index(const size_t pos) const {
    std::vector<size_t> ind(this->shape.size(), 0);
    auto pos_copy = pos;
    for (int i = (int)this->shape.size() - 1; i >= 0; --i) {
      ind[i] = pos_copy % this->shape[i];
      pos_copy /= this->shape[i];
    }
    return ind;
  }

  /**
   * @brief Scalar and diag. vector of Dense format matrix add
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_add(const Float alpha);

  /**
   * @brief Scalar and diag. vector of Dense format matrix sub
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_sub(const Float alpha);

  /**
   * @brief Scalar and diag. vector of Dense format matrix mul
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_mul(const Float alpha);

  /**
   * @brief Scalar and diag. vector of Dense format matrix div
   * @param alpha scalar
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_div(const Float alpha);

  /**
   * @brief Vector and diag. vector of Dense format matrix add
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_add(const vector<Float> &vec);
  void diag_add(const view1D<vector<Float>, Float> &vec);
  void diag_add(const view1D<matrix::Dense<Float>, Float> &vec);
  void diag_add(const view1D<tensor::tensor_Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of Dense format matrix sub
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_sub(const vector<Float> &vec);
  void diag_sub(const view1D<vector<Float>, Float> &vec);
  void diag_sub(const view1D<matrix::Dense<Float>, Float> &vec);
  void diag_sub(const view1D<tensor::tensor_Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of Dense format matrix mul
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_mul(const vector<Float> &vec);
  void diag_mul(const view1D<vector<Float>, Float> &vec);
  void diag_mul(const view1D<matrix::Dense<Float>, Float> &vec);
  void diag_mul(const view1D<tensor::tensor_Dense<Float>, Float> &vec);

  /**
   * @brief Vector and diag. vector of Dense format matrix div
   * @param vec vector
   * @note
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  void diag_div(const vector<Float> &vec);
  void diag_div(const view1D<vector<Float>, Float> &vec);
  void diag_div(const view1D<matrix::Dense<Float>, Float> &vec);
  void diag_div(const view1D<tensor::tensor_Dense<Float>, Float> &vec);
};
/**@}*/

} // namespace monolish
