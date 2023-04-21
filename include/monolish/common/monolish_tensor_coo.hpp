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
  std::shared_ptr<Float> vad;

  /**
   * @brief # of non-zero element
   */
  size_t vad_nnz = 0;

  /**
   * @brief alloced matrix size
   */
  std::size_t alloc_nnz = 0;

  /**
   * @brief matrix create flag;
   */
  bool vad_create_flag = false;

  tensor_COO() : shape(), gpu_status(false), index(), vad_nnz(0) {
    vad_create_flag = true;
  }

  tensor_COO(const std::vector<size_t> &shape_)
      : shape(shape_), gpu_status(false), index(), vad_nnz(0) {
    vad_create_flag = true;
  }

  void convert(const tensor::tensor_Dense<Float> &tens);

  tensor_COO(const tensor::tensor_Dense<Float> &tens) {
    vad_create_flag = true;
    convert(tens);
  }

  tensor_COO(const std::vector<size_t> &shape_,
             const std::vector<std::vector<size_t>> &index_,
             const Float *value);

  tensor_COO(const tensor_COO<Float> &coo);

  tensor_COO(const tensor_COO<Float> &coo, Float value);

  void print_all(bool force_cpu = false) const;

  void print_all(const std::string filename) const;

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
   * @brief Get matrix element (A(i,j))
   * @note
   * - # of computation: i*M+j
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const std::vector<size_t> &pos) const;

  /**
   * @brief Get matrix element (A(i,j))
   * @note
   * - # of computation: i*M+j
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const std::vector<size_t> &pos) {
    return static_cast<const tensor_COO *>(this)->at(pos);
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
   * @brief get # of non-zeros
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] size_t get_nnz() const { return vad_nnz; }

  void set_shape(const std::vector<size_t> &shape) { this->shape = shape; }

  [[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }

  [[nodiscard]] const Float *data() const { return vad.get(); }

  [[nodiscard]] Float *data() { return vad.get(); }

  void resize(const size_t N, Float val = 0) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU matrix cant use resize");
    }
    if (vad_create_flag) {
      std::shared_ptr<Float> tmp(new Float[N], std::default_delete<Float[]>());
      size_t copy_size = std::min(vad_nnz, N);
      for (size_t i = 0; i < copy_size; ++i) {
        tmp.get()[i] = vad.get()[i];
      }
      for (size_t i = copy_size; i < N; ++i) {
        tmp.get()[i] = val;
      }
      vad = tmp;
      alloc_nnz = N;
      vad_nnz = N;

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

  [[nodiscard]] bool equal(const tensor_COO<Float> &tens,
                           bool compare_cpu_and_device = false) const;

  [[nodiscard]] bool operator==(const tensor_COO<Float> &tens) const;

  [[nodiscard]] bool operator!=(const tensor_COO<Float> &tens) const;

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

  std::vector<size_t> get_index(const size_t pos) {
    std::vector<size_t> ind(this->shape.size(), 0);
    auto pos_copy = pos;
    for (int i = this->shape.size() - 1; i >= 0; --i) {
      ind[i] = pos_copy % this->shape[i];
      pos_copy /= this->shape[i];
    }
    return ind;
  }

  void insert(const std::vector<size_t> &pos, const Float val);

private:
  void _q_sort(int lo, int hi);

public:
  void sort(bool merge);
};
} // namespace tensor
} // namespace monolish
