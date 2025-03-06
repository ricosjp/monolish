#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_Dense;
template <typename Float> class tensor_COO;
template <typename Float> class tensor_CRS {
private:
  std::vector<size_t> shape;

  mutable bool gpu_status = false;

  size_t structure_hash;

  size_t first = 0;

public:
  std::shared_ptr<Float> val;

  size_t val_nnz = 0;

  size_t alloc_nnz = 0;

  bool val_create_flag = false;

  std::vector<std::vector<int>> row_ptrs;

  std::vector<std::vector<int>> col_inds;

  tensor_CRS()
      : shape(), gpu_status(false), row_ptrs(), col_inds(), val_nnz(0) {
    val_create_flag = true;
  }

  tensor_CRS(const std::vector<size_t> &shape_)
      : shape(shape_), gpu_status(false), row_ptrs(), col_inds(), val_nnz(0) {
    val_create_flag = true;
  }

  tensor_CRS(const std::initializer_list<size_t> &shape_)
      : shape(shape_), gpu_status(false), row_ptrs(), col_inds(), val_nnz(0) {
    val_create_flag = true;
  }

  void convert(const tensor::tensor_COO<Float> &coo);

  tensor_CRS(const tensor::tensor_COO<Float> &coo) {
    val_create_flag = true;
    convert(coo);
  }

  void convert(const tensor::tensor_CRS<Float> &crs);

  tensor_CRS(const tensor::tensor_CRS<Float> &crs) {
    val_create_flag = true;
    convert(crs);
  }

  void convert(const matrix::CRS<Float> &crs);

  tensor_CRS(const matrix::CRS<Float> &crs) {
    val_create_flag = true;
    convert(crs);
  }

  tensor_CRS(const std::vector<size_t> &shape_,
             const std::vector<std::vector<int>> &row_ptrs_,
             const std::vector<std::vector<int>> &col_inds_,
             const Float *value);

  tensor_CRS(const tensor_CRS<Float> &crs, Float value);

  void print_all(bool force_cpu = false) const;

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: size
   **/
  void send() const;

  /**
   * @brief recv. data to GPU, and free data on GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: size
   **/
  void recv();

  /**
   * @brief recv. data to GPU (w/o free)
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: size
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

  // TODO
  /**
   * @brief Memory data space required by the matrix
   * @note
   * - # of computation: 3
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] double get_data_size() const {
    return get_nnz() * sizeof(Float) / 1.0e+9;
  }

  void set_ptr(const std::vector<size_t> &shape,
               const std::vector<std::vector<int>> &rowptrs,
               const std::vector<std::vector<int>> &colinds,
               const std::vector<Float> &value);

  void set_ptr(const std::vector<size_t> &shape,
               const std::vector<std::vector<int>> &rowptrs,
               const std::vector<std::vector<int>> &colinds, const size_t vsize,
               const Float *value);

  void set_ptr(const std::vector<size_t> &shape,
               const std::vector<std::vector<int>> &rowptrs,
               const std::vector<std::vector<int>> &colinds, const size_t vsize,
               const Float value);

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
  [[nodiscard]] std::shared_ptr<Float> get_val() { return val; }

  /**
   * @brief get # of non-zeros
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_nnz() const { return val_nnz; }

  /**
   * @brief get # of alloced non-zeros
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_alloc_nnz() const { return alloc_nnz; }

  /**
   * @brief get first position
   * @return first position
   * @note
   * - # of computation: 1
   */
  [[nodiscard]] size_t get_first() const { return first; }

  /**
   * @brief get first position (same as get_first())
   * @return first position
   * @note
   * - # of computation: 1
   */
  [[nodiscard]] size_t get_offset() const { return get_first(); }

  /**
   * @brief change first position
   * @note
   * - # of computation: 1
   */
  void set_first(size_t i) { first = i; }

  /**
   * @brief Set shape
   * @param shape shape of tensor
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_shape(const std::vector<size_t> &shape) { this->shape = shape; }

  /**
   * @brief true: sended, false: not send
   * @return gpu status
   * **/
  [[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }

  ~tensor_CRS() {
    if (val_create_flag) {
      if (get_device_mem_stat()) {
        device_free();
      }
    }
  }

  /**
   * @brief returns a direct pointer to the tensor
   * @return A const pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *data() const { return val.get(); }

  /**
   * @brief returns a direct pointer to the tensor
   * @return A pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *data() { return val.get(); }

  /**
   * @brief resize tensor value
   * @param N tensor size
   * @note
   * - # of computation: N
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void resize(const size_t N, Float Val = 0) {
    if (first + N < alloc_nnz) {
      for (size_t i = val_nnz; i < N; ++i) {
        begin()[i] = Val;
      }
      val_nnz = N;
      return;
    }
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
      first = 0;
    } else {
      throw std::runtime_error("Error, not create vector cant use resize");
    }
  }

  /**
   * @brief get format name "tensor_CRS"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const { return "tensor_CRS"; }

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

  /**
   * @brief returns a begin iterator
   * @return begin iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *begin() const { return data() + get_offset(); }

  /**
   * @brief returns a begin iterator
   * @return begin iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *begin() { return data() + get_offset(); }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *end() const {
    return data() + get_offset() + get_nnz();
  }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *end() { return data() + get_offset() + get_nnz(); }

  /**
   * @brief fill tensor elements with a scalar value
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
  void operator=(const tensor_CRS<Float> &mat);

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
    return data()[first + i];
  }

  /**
   * @brief Comparing matrices (A == mat)
   * @param mat CRS matrix
   * @param compare_cpu_and_device compare data on both CPU and GPU
   * @return true or false
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  [[nodiscard]] bool equal(const tensor_CRS<Float> &mat,
                           bool compare_cpu_and_device = false) const;

  /**
   * @brief Comparing matrices (A == mat)
   * @param mat CRS matrix
   * @return true or false
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator==(const tensor_CRS<Float> &mat) const;

  /**
   * @brief Comparing matrices (A != mat)
   * @param mat CRS matrix
   * @return true or false
   * @note
   * - # of computation: (M+1)+2nnz
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator!=(const tensor_CRS<Float> &mat) const;
};

} // namespace tensor

} // namespace monolish
