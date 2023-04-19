#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"

namespace monolish {
// template <typename Float> class vector;
// template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_Dense {
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
   * @brief Dense tensor format value (pointer)
   **/
  std::shared_ptr<Float> vad;

  /**
   * @brief # of non-zero element (M * N)
   **/
  size_t vad_nnz = 0;

  /**
   * @brief allocated tensor size
   **/
  size_t alloc_nnz = 0;

  /**
   * @brief tensor create flag
   **/
  bool vad_create_flag = false;

  tensor_Dense() { vad_create_flag = true; }

  /**
   * @brief create Dense tensor from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const matrix::Dense<Float> &dense);

  /**
   * @brief create Dense tensor from vector
   * @param vec input vector (size M)
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const vector<Float> &vec);

  /**
   * @brief Allocate dense tensor
   * @param shape shape of tensor
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  tensor_Dense(const std::vector<size_t> &shape);

  /**
   * @brief Allocate dense tensor
   * @param shape shape of tensor
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  tensor_Dense(const std::vector<size_t> &shape, const Float *value);

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

  /**
   * @brief Set row number
   * @param N # of row
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_shape(const std::vector<size_t> &shape) { reshape(shape); };

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: M*N
   **/
  void send() const;

  /**
   * @brief recv. data to GPU, and free data on GPU
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: M*N
   **/
  void recv();

  /**
   * @brief recv. data to GPU (w/o free)
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: M*N
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

  /**
   * @brief true: sended, false: not send
   * @return gpu status
   * **/
  [[nodiscard]] bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief destructor of dense tensor, free GPU memory
   * @note
   * - Multi-threading: false
   * - GPU acceleration: true
   *    - # of data transfer: 0
   * **/
  ~tensor_Dense() {
    if (vad_create_flag) {
      if (get_device_mem_stat()) {
        device_free();
      }
    }
  }

  /**
   * @brief returns a direct pointer to the matrix
   * @return A const pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] const Float *data() const { return vad.get(); }

  /**
   * @brief returns a direct pointer to the tensor
   * @return A pointer to the first element
   * @note
   * - # of computation: 1
   **/
  [[nodiscard]] Float *data() { return vad.get(); }

  /**
   * @brief resize tensor value
   * @param N tensor size
   * @note
   * - # of computation: N
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void resize(size_t N, Float val = 0) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU matrix cant use resize");
    }
    if (N == 0) {
      throw std::runtime_error("Error, tensor must have at least 1 element");
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
    } else {
      throw std::runtime_error("Error, not create vector cant use resize");
    }
  }

  void resize(std::vector<size_t> &shape, Float val = 0) {
    size_t N = 1;
    for (auto n : shape) {
      N *= n;
    }
    resize(N, val);
    this->shape = shape;
  }

  /**
   * @brief move Dense tensor from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * - # of computation: 1
   */
  void move(const matrix::Dense<Float> &dense);

  /**
   * @brief move Dense tensor from vector
   * @param vec input vector (size M)
   * - # of computation: 1
   */
  void move(const vector<Float> &vec);

  /**
   * @brief Reshape tensor
   * @param shape
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void reshape(const std::vector<size_t> &shape);
};
} // namespace tensor
} // namespace monolish
