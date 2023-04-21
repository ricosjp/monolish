#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
namespace tensor {
template <typename Float> class tensor_COO;
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

  void convert(const tensor::tensor_Dense<Float> &tens);

  void convert(const tensor::tensor_COO<Float> &tens);

  tensor_Dense(const tensor::tensor_COO<Float> &tens) {
    convert(tens);
    vad_create_flag = true;
  }

  /**
   * @brief create Dense tensor from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const matrix::Dense<Float> &dense);

  tensor_Dense(const matrix::Dense<Float> &dense) {
    convert(dense);
    vad_create_flag = true;
  }

  /**
   * @brief create Dense tensor from vector
   * @param vec input vector (size M)
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const vector<Float> &vec);

  tensor_Dense(const vector<Float> &vec) {
    convert(vec);
    vad_create_flag = true;
  }

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
   * @brief Allocate dense tensor
   * @param shape shape of tensor
   * @note
   * - # of computation: nnz
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  tensor_Dense(const std::vector<size_t> &shape,
               const std::vector<Float> &value);

  tensor_Dense(const std::vector<size_t> &shape, const Float min,
               const Float max);

  tensor_Dense(const std::vector<size_t> &shape, const Float min,
               const Float max, const std::uint32_t seed);

  tensor_Dense(const tensor_Dense<Float> &tens);

  tensor_Dense(const tensor_Dense<Float> &tens, Float value);

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
  void set_shape(const std::vector<size_t> &shape) { this->shape = shape; };

  /**
   * @brief get format name "tensor_Dense"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string type() const { return "tensor_Dense"; }

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

  [[nodiscard]] Float at(const std::vector<size_t> &pos) const;

  [[nodiscard]] Float at(const std::vector<size_t> &pos) {
    return static_cast<const tensor_Dense *>(this)->at(pos);
  };

  void insert(const std::vector<size_t> &pos, const Float Val);

  void print_all(bool force_cpu = false) const;

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
    } else {
      throw std::runtime_error("Error, not create vector cant use resize");
    }
  }

  void resize(const std::vector<size_t> &shape, Float val = 0) {
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

  [[nodiscard]] bool equal(const tensor_Dense<Float> &tens,
                           bool compare_cpu_and_device = false) const;

  [[nodiscard]] bool operator==(const tensor_Dense<Float> &tens) const;

  [[nodiscard]] bool operator!=(const tensor_Dense<Float> &tens) const;

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
   * @brief Reshape tensor
   * @param shape
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void reshape(const std::vector<size_t> &shape);

  /////////////////////////////////////////////

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
};
} // namespace tensor
} // namespace monolish
