/**
 * @autor RICOS Co. Ltd.
 * @file monolish_vector.h
 * @brief declare vector class
 * @date 2019
 **/

#pragma once
#include "./monolish_logger.hpp"
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
//#typedef typename std::allocator_trails::allocator_type::reference reference;

namespace monolish {

/**
 * @brief vector type
 * @note
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): false
 */
template <typename Float> class vector {
private:
  /**
   * @brief size N vector data
   **/
  std::vector<Float> val;

  /**
   * @brief true: sended, false: not send
   **/
  mutable bool gpu_status = false;

public:
  vector() {}

  // constractor ///////////////////////////////////////////////////////
  /**
   * @brief allocate size N vector
   * @param N vector length
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  //vector(const size_t N) { val.resize(N); }
  vector(const size_t N);

  /**
   * @brief initialize size N vector, value to fill the container
   * @param N vector length
   * @param value fill Float type value to all elements
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  vector(const size_t N, const Float value);

  /**
   * @brief copy from std::vector
   * @param vec input std::vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  vector(const std::vector<Float> &vec);

  /**
   * @brief copy from monolish::vector
   * @param vec input monolish::vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: N (allocation)
   *        - if `vec.gpu_statius == true`; coping data only on GPU
   *        - else; coping data only on CPU
   **/
  vector(const vector<Float> &vec);

  /**
   * @brief copy from pointer
   * @param start start pointer
   * @param end  end pointer
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  vector(const Float *start, const Float *end);

  /**
   * @brief create N length rand(min~max) vector
   * @param N vector length
   * @param min rand min
   * @param max rand max
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): false
   **/
  vector(const size_t N, const Float min, const Float max);

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   * @note
   * - # of data transfer: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   **/
  void send() const;

  /**
   * @brief recv data from GPU, and free data on GPU
   * @note
   * - # of data transfer: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   **/
  void recv();

  /**
   * @brief recv data from GPU (w/o free)
   * @note
   * - # of data transfer: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   **/
  void nonfree_recv();

  /**
   * @brief free data on GPU
   * @note
   * - # of data transfer: 0
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   **/
  void device_free() const;

  /**
   * @brief true: sended, false: not send
   * @return gpu status
   * @note
   * - # of data transfer: 0
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   **/
  bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief destructor of vector, free GPU memory
   * @note
   * - # of data transfer: 0
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   **/
  ~vector() {
    if (get_device_mem_stat()) {
      device_free();
    }
  }

  // util
  // ///////////////////////////////////////////////////////////////////////////

  /**
   * @brief returns a direct pointer to the vector
   * @return A pointer to the first element
   * @note
   * - # of computation: 1
   **/
  Float *data() { return val.data(); }

  /**
   * @brief returns a direct pointer to the vector
   * @return A const pointer to the first element
   * @note
   * - # of computation: 1
   **/
  const Float *data() const { return val.data(); }

  /**
   * @brief resize vector (only CPU)
   * @param N vector length
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void resize(size_t N) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    val.resize(N);
  }

  /**
   * @brief Add a new element at the ent of the vector (only CPU)
   * @param val new element
   * @note
   * - # of computation: 1
   **/
  void resize(Float val) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    val.push_back(val);
  }

  /**
   * @brief returns a begin iterator
   * @return begin iterator
   * @note
   * - # of computation: 1
   **/
  auto begin() { return val.begin(); }

  /**
   * @brief returns a end iterator
   * @return end iterator
   * @note
   * - # of computation: 1
   **/
  auto end() { return val.end(); }

  /**
   * @brief get vector size
   * @return vector size
   * @note
   * - # of computation: 1
   **/
  auto size() const { return val.size(); }

  /**
   * @brief get vector size
   * @return vector size
   * @note
   * - # of computation: 1
   **/
  auto get_nnz() const { return val.size(); }

  /**
   * @brief vector copy ( Copy the memory data on CPU and GPU )
   * @return copied vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: N (allocation)
   *        - if `vec.gpu_statius == true`; copy on CPU; then send to GPU
   *        - else; coping data only on CPU
   **/
  vector copy();

  /**
   * @brief print all elements to standart I/O
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void print_all() const;

  /**
   * @brief print all elements to file
   * @param filename output filename
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void print_all(std::string filename) const;

  // operator
  // ///////////////////////////////////////////////////////////////////////////

  /**
   * @brief copy vector, It is same as copy ( Copy the memory on CPU and GPU )
   * @param vec source vector
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): true
   *    - # of data transfer: N (allocation)
   *        - if `vec.gpu_statius == true`; copy on CPU
   *        - else; coping data on CPU
   **/
  void operator=(const vector<Float> &vec);

  /**
   * @brief copy vector from std::vector
   * @param vec source std::vector
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  void operator=(const std::vector<Float> &vec);

  /**
   * @brief Sign inversion
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  vector<Float> operator-();

  // vec - scalar

  /**
   * @brief add scalar to vector elements (v = v[0:N] + value)
   * @param value source value
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   * @warning
   * The arithmetic operator need to allocate tmp. array (size N)
   **/
  vector<Float> operator+(const Float value);

  /**
   * @brief sub scalar to vector elements (v = v[0:N] - value)
   * @param value source value
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   * @warning
   * The arithmetic operator need to allocate tmp. array (size N)
   **/
  vector<Float> operator-(const Float value);

  /**
   * @brief mul scalar to vector elements (v = v[0:N] * value)
   * @param value source value
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   * @warning
   * The arithmetic operator need to allocate tmp. array (size N)
   **/
  vector<Float> operator*(const Float value);

  /**
   * @brief div scalar to vector elements (v = v[0:N] / value)
   * @param value source value
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   * @warning
   * The arithmetic operator need to allocate tmp. array (size N)
   **/
  vector<Float> operator/(const Float value);

  /**
   * @brief add scalar to vector elements (v[0:N] += value)
   * @param value source value
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void operator+=(const Float value);

  /**
   * @brief sub scalar to vector elements (v[0:N] += value)
   * @param value source value
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void operator-=(const Float value);
  /**
   * @brief mul scalar to vector elements (v[0:N] += value)
   * @param value source value
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void operator*=(const Float value);

  /**
   * @brief div scalar to vector elements (v[0:N] += value)
   * @param value source value
   * @return output vector
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void operator/=(const Float value);

  // vec - vec

  /**
   * @brief add vector elements (v[0:N] = v[0:N] + vec[0:N])
   * @param vec vector (size N)
   * @return output vector (size N)
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   * @warning
   * The arithmetic operator need to allocate tmp. array (size N)
   **/
  vector<Float> operator+(const vector<Float> &vec);

  /**
   * @brief sub vector elements (v[0:N] = v[0:N] - vec[0:N])
   * @param vec vector (size N)
   * @return output vector (size N)
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   * @warning
   * The arithmetic operator need to allocate tmp. array (size N)
   **/
  vector<Float> operator-(const vector<Float> &vec);

  /**
   * @brief mul vector elements (v[0:N] = v[0:N] + vec[0:N])
   * @param vec vector (size N)
   * @return output vector (size N)
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   * @warning
   * The arithmetic operator need to allocate tmp. array (size N)
   **/
  vector<Float> operator*(const vector<Float> &vec);

  /**
   * @brief div vector elements (v[0:N] = v[0:N] + vec[0:N])
   * @param vec vector (size N)
   * @return output vector (size N)
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   * @warning
   * The arithmetic operator need to allocate tmp. array (size N)
   **/
  vector<Float> operator/(const vector<Float> &vec);

  /**
   * @brief add vector elements (v[0:N] += vec[0:N])
   * @param vec vector (size N)
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void operator+=(const vector<Float> &vec);

  /**
   * @brief sub vector elements (v[0:N] -= vec[0:N])
   * @param vec vector (size N)
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void operator-=(const vector<Float> &vec);

  /**
   * @brief mul vector elements (v[0:N] *= vec[0:N])
   * @param vec vector (size N)
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void operator*=(const vector<Float> &vec);

  /**
   * @brief div vector elements (v[0:N] /= vec[0:N])
   * @param vec vector (size N)
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void operator/=(const vector<Float> &vec);

  /**
   * @brief refetrence to the element at position (v[i])
   * @param i Position of an element in the vector
   * @return vector element (v[i])
   * @note
   * - # of computation: 1
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  Float &operator[](size_t i) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    return val[i];
  }

  /**
   * @brief Comparing vectors (v == vec)
   * @param vec vector (size N)
   * @return true or false
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  bool operator==(const vector<Float> &vec) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator==");
    }
    if (val.size() != vec.size())
      return false;
    for (size_t i = 0; i < vec.size(); i++) {
      if (val[i] != vec.val[i])
        return false;
    }
    return true;
  }

  /**
   * @brief Comparing vectors (v != vec)
   * @param vec vector (size N)
   * @return true or false
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): false
   * - GPU acceleration (OpenACC): false
   **/
  bool operator!=(const vector<Float> &vec) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator!=");
    }
    if (val.size() != vec.size())
      return true;
    for (size_t i = 0; i < vec.size(); i++) {
      if (val[i] != vec.val[i])
        return true;
    }
    return false;
  }

  /**
   * @brief tanh vector elements (v[0:N] = tanh(vec[0:N]))
   * @note
   * - # of computation: N
   * - Multi-threading (OpenMP): true
   * - GPU acceleration (OpenACC): true
   **/
  void tanh();
};
} // namespace monolish
