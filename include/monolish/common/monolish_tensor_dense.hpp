#pragma once
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"

namespace monolish {
template <typename Float> class vector;
template <typename TYPE, typename Float> class view1D;
template <typename TYPE, typename Float> class view_Dense;
template <typename TYPE, typename Float> class view_tensor_Dense;
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
  std::shared_ptr<Float> val;

  /**
   * @brief # of non-zero element (M * N)
   **/
  size_t val_nnz = 0;

  /**
   * @brief allocated tensor size
   **/
  size_t alloc_nnz = 0;

  /**
   * @brief tensor create flag
   **/
  bool val_create_flag = false;

  tensor_Dense() { val_create_flag = true; }

  /**
   * @brief Create tensor_Dense tensor from tensor_Dense tensor
   * @param tens input tensor_Dense tensor
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const tensor::tensor_Dense<Float> &tens);

  /**
   * @brief Create tensor_Dense tensor from tensor_Dense tensor
   * @param tens tensor_Dense format tensor
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: size (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  tensor_Dense(const tensor::tensor_Dense<Float> &tens);

  /**
   * @brief Create tensor_Dense tensor from tensor_COO tensor
   * @param tens input tensor_Dense tensor
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const tensor::tensor_COO<Float> &tens);

  /**
   * @brief Create tensor_Dense tensor from tensor_COO tensor
   * @param tens input tensor_COO tensor
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  tensor_Dense(const tensor::tensor_COO<Float> &tens) {
    convert(tens);
    val_create_flag = true;
  }

  /**
   * @brief create tensor_Dense tensor from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const matrix::Dense<Float> &dense);

  /**
   * @brief Create tensor_Dense tensor from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  tensor_Dense(const matrix::Dense<Float> &dense) {
    convert(dense);
    val_create_flag = true;
  }

  /**
   * @brief create tensor_Dense tensor from vector
   * @param vec input vector (size M)
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  void convert(const vector<Float> &vec);

  /**
   * @brief create tensor_Dense tensor from vector
   * @param vec input vector (size M)
   * - # of computation: M
   * - Multi-threading: true
   * - GPU acceleration: false
   **/
  tensor_Dense(const vector<Float> &vec) {
    convert(vec);
    val_create_flag = true;
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
  tensor_Dense(const std::initializer_list<size_t> &shape);

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
   * @brief Allocate tensor_Dense tensor
   * @param shape shape of tensor
   * @param value value std::vector
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  tensor_Dense(const std::vector<size_t> &shape,
               const std::vector<Float> &value);

  /**
   * @brief Allocate tensor_Dense tensor
   * @param shape shape of tensor
   * @param min rand min
   * @param max rand max
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  tensor_Dense(const std::vector<size_t> &shape, const Float min,
               const Float max);

  /**
   * @brief Allocate tensor_Dense tensor
   * @param shape shape of tensor
   * @param min rand min
   * @param max rand max
   * @param seed random seed
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  tensor_Dense(const std::vector<size_t> &shape, const Float min,
               const Float max, const std::uint32_t seed);

  /**
   * @brief Create tensor_Dense tensor of the same size as input tensor
   * @param tens input tensor_Dense tensor
   * @param value the value to initialize elements
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: size (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  tensor_Dense(const tensor_Dense<Float> &tens, Float value);

  /**
   * @brief Create Dense matrix from view Dense matrix
   * @param dense Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M*N (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  tensor_Dense(const view_tensor_Dense<vector<Float>, Float> &tens);

  /**
   * @brief Create Dense matrix from view Dense matrix
   * @param dense Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M*N (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  tensor_Dense(const view_tensor_Dense<matrix::Dense<Float>, Float> &tens);

  /**
   * @brief Create Dense matrix from view Dense matrix
   * @param dense Dense format matrix
   * @note
   * - # of computation: M*N
   * - Multi-threading: true
   * - GPU acceleration: true
   *    - # of data transfer: M*N (only allocation)
   *        - if `dense.gpu_status == true`; coping data on CPU and GPU
   *respectively
   *        - else; coping data only on CPU
   **/
  tensor_Dense(
      const view_tensor_Dense<tensor::tensor_Dense<Float>, Float> &tens);

  /**
   * @brief Set tensor_Dense array from std::vector
   * @param shape shape of tensor
   * @param value value (size nnz)
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const std::vector<size_t> &shape,
               const std::vector<Float> &value);

  /**
   * @brief Set tensor_Dense array from array
   * @param shape shape of tensor
   * @param value value (size nnz)
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void set_ptr(const std::vector<size_t> &shape, const Float *value);

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
   * @brief Set shape
   * @param shape shape of tensor
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
   * @brief get element A[index]...
   * @param pos aligned position index
   * @return A[index]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t pos) const;

  /**
   * @brief get element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @return A[pos[0]][pos[1]]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const std::vector<size_t> &pos) const;

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
#if !defined(__clang__) && defined(__GNUC__)
  [[nodiscard]] Float at(const size_t dim, const size_t dim2,
                         const Args... args) const {
    std::vector<size_t> pos(1);
    pos[0] = dim;
    return this->at(pos, dim2, args...);
  };
#else
  [[nodiscard]] Float at(const size_t dim, const Args... args) const {
    std::vector<size_t> pos(1);
    pos[0] = dim;
    return this->at(pos, args...);
  };
#endif

  /**
   * @brief get element A[index]... (onlu CPU)
   * @param pos aligned position index
   * @return A[index]...
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] Float at(const size_t pos) {
    return static_cast<const tensor_Dense *>(this)->at(pos);
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
  [[nodiscard]] Float at(const std::vector<size_t> &pos) {
    return static_cast<const tensor_Dense *>(this)->at(pos);
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
    return static_cast<const tensor_Dense *>(this)->at(pos, args...);
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
    return static_cast<const tensor_Dense *>(this)->at(dim, args...);
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
  void insert(const size_t pos, const Float Val);

  /**
   * @brief set element A[pos[0]][pos[1]]...
   * @param pos std::vector position
   * @param Val scalar value
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void insert(const std::vector<size_t> &pos, const Float Val);

  /**
   * @brief print all elements to standard I/O
   * @param force_cpu Ignore device status and output CPU data
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
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
    } else {
      throw std::runtime_error("Error, not create vector cant use resize");
    }
  }

  /**
   * @brief resize tensor value
   * @param shape tensor shape
   * @note
   * - # of computation: size
   * - Multi-threading: false
   * - GPU acceleration: false
   */
  void resize(const std::vector<size_t> &shape, Float val = 0) {
    size_t N = 1;
    for (auto n : shape) {
      N *= n;
    }
    resize(N, val);
    this->shape = shape;
  }

  /**
   * @brief move tensor_Dense tensor from Dense matrix
   * @param dense input Dense matrix (size M x N)
   * - # of computation: 1
   */
  void move(const matrix::Dense<Float> &dense);

  /**
   * @brief move tensor_Dense tensor from vector
   * @param vec input vector (size M)
   * - # of computation: 1
   */
  void move(const vector<Float> &vec);

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
   * @brief fill tensor elements with a scalar value
   * @param value scalar value
   * @note
   * - # of computation: size
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
  void operator=(const tensor_Dense<Float> &tens);

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
   * @param tens tensor_Dense tensor
   * @param compare_cpu_and_device compare data on both CPU and GPU
   * @return true or false
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   **/
  [[nodiscard]] bool equal(const tensor_Dense<Float> &tens,
                           bool compare_cpu_and_device = false) const;

  /**
   * @brief Comparing tensors (A == tens)
   * @param tens tensor_Dense tensor
   * @return true or false
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator==(const tensor_Dense<Float> &tens) const;

  /**
   * @brief Comparing tensors (A != tens)
   * @param tens tensor_Dense tensor
   * @return true or false
   * @note
   * - # of computation: size
   * - Multi-threading: true
   * - GPU acceleration: true
   *   - if `gpu_status == true`; compare data on GPU
   *   - else; compare data on CPU
   **/
  [[nodiscard]] bool operator!=(const tensor_Dense<Float> &tens) const;

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
   * @brief Reshape tensor
   * @param shape
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  void reshape(const std::vector<int> &shape);

  /**
   * @brief Reshape tensor
   * @param shape
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  template <typename... Args>
  void reshape(const std::vector<int> &shape, const size_t dim,
               const Args... args) {
    std::vector<int> shape_copy = shape;
    shape_copy.push_back(dim);
    reshape(shape_copy, args...);
    return;
  }

  /**
   * @brief Reshape tensor
   * @param shape
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  template <typename... Args> void reshape(const int dim, const Args... args) {
    std::vector<int> shape(1);
    shape[0] = dim;
    reshape(shape, args...);
    return;
  }

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
} // namespace tensor
} // namespace monolish
