#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_Dense;
template <typename Float> class tensor_COO {
private:
  /**
   * @brief shape
   **/
  std::vector<size_t> shape;

  /**
   * @brief true: sended, false: not send
   **/
  mutable bool gpu_status = false;

public:
  /**
   * @brief Coodinate format index, which stores index numbers of the non-zero
   * elements (size nnz)
   */
  std::vector<std::vector<size_t>> index;

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

  tensor_COO() : shape(), gpu_status(false), index(), val_nnz(0) {
    val_create_flag = true;
  }

  /**
   * @brief Initialize tensor_COO tensor
   * @param shape shape of tensor
   * @note
   * - # of computation: 0
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  tensor_COO(const std::vector<size_t> &shape_)
      : shape(shape_), gpu_status(false), index(), val_nnz(0) {
    val_create_flag = true;
  }

  /**
   * @brief Create tensor_COO tensor from tensor_Dense tensor
   * @param tens input tensor_Dense tensor
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void convert(const tensor::tensor_Dense<Float> &tens);

  /**
   * @brief Create tensor_COO tensor from tensor_Dense tensor
   * @param tens input tensor_Dense tensor
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  tensor_COO(const tensor::tensor_Dense<Float> &tens) {
    val_create_flag = true;
    convert(tens);
  }

  /**
   * @brief Create tensor_COO tensor from n-origin array
   * @param shape_ shape of tensor
   * @param index_ n-origin index, which stores the numbers of the non-zero
   *elements (size nnz)
   * @param value n-origin value, which stores the non-zero elements (size nnz)
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  tensor_COO(const std::vector<size_t> &shape_,
             const std::vector<std::vector<size_t>> &index_,
             const Float *value);

  /**
   * @brief Create tensor_COO tensor from tensor_COO tensor
   * @param tens input tensor_COO tensor
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  tensor_COO(const tensor_COO<Float> &coo);

  /**
   * @brief Initialize tensor_COO tensor of the same size as input tensor
   * @param tens input tensor_COO tensor
   * @param value the value to initialize elements
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  tensor_COO(const tensor_COO<Float> &coo, Float value);

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

  /**
   * @brief get element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const std::vector<size_t> &pos) const;

  /**
   * @brief get element A[pos[0]][pos[1]]... (onlu CPU)
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const std::vector<size_t> &pos) {
    return static_cast<const tensor_COO *>(this)->at(pos);
  };

  /**
   * @brief Set tensor_COO array from std::vector
   * @param shape shape of tensor
   * @param indix index fo tensor
   * @param v value
   * @note
   * - # of computation: 3
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const std::vector<size_t> &shape,
               const std::vector<std::vector<size_t>> &index,
               const std::vector<Float> &v);

  /**
   * @brief Set tensor_COO array from array
   * @param shape shape of tensor
   * @param indix index fo tensor
   * @param vsize size of value
   * @param v value
   * @note
   * - # of computation: 3
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const std::vector<size_t> &shape,
               const std::vector<std::vector<size_t>> &index,
               const size_t vsize, const Float *v);

  /**
   * @brief get shape
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::vector<size_t> get_shape() const { return shape; }

  /**
   * @brief get # of non-zeros
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_nnz() const { return val_nnz; }

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

      index.resize(N);
    } else {
      throw std::runtime_error("Error, not create vector cant use resize");
    }
  }

  /**
   * @brief get format name "tensor_COO"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const { return "tensor_COO"; }

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

  /**
   * @brief tensor copy
   * @param tens COO tensor
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   * @warning
   * src. and dst. must be same non-zero structure (dont check in this function)
   **/
  void operator=(const tensor_COO<Float> &tens);

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
   * @brief Comparing tensors (A == tens)
   * @param tens tensor_COO tensor
   * @param compare_cpu_and_device compare data on both CPU and GPU
   * @return true or false
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  [[nodiscard]] bool equal(const tensor_COO<Float> &tens,
                           bool compare_cpu_and_device = false) const;

  /**
   * @brief Comparing tensors (A == tens)
   * @param tens tensor_COO tensor
   * @return true or false
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator==(const tensor_COO<Float> &tens) const;

  /**
   * @brief Comparing tensors (A != tens)
   * @param tens tensor_COO tensor
   * @return true or false
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator!=(const tensor_COO<Float> &tens) const;

  /**
   * @brief get aligned index from vector index (A[pos] = A[ind[0]][ind[1]]...)
   * @param pos position (std::vector)
   * @return aligned position
   * @note
   * - # of computation: shape size
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  size_t get_index(const std::vector<size_t> &pos) {
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
  std::vector<size_t> get_index(const size_t pos) {
    std::vector<size_t> ind(this->shape.size(), 0);
    auto pos_copy = pos;
    for (int i = (int)this->shape.size() - 1; i >= 0; --i) {
      ind[i] = pos_copy % this->shape[i];
      pos_copy /= this->shape[i];
    }
    return ind;
  }

  /**
   * @brief insert element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @param val scalar value
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void insert(const std::vector<size_t> &pos, const Float val);

private:
  void _q_sort(int lo, int hi);

public:
  /**
   * @brief sort tensor_COO tensor elements (and merge elements)
   * @param merge need to merge (true or false)
   * @note
   * - # of computation: 3nnz x log(3nnz) ~ 3nnz^2
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void sort(bool merge);
};
} // namespace tensor
} // namespace monolish
