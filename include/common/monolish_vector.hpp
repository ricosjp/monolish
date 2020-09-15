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
 * @brief std::vector-like vector class
 */
template <typename Float> class vector {
private:
  std::vector<Float> val;
  mutable bool gpu_status = false; // true: sended, false: not send

public:
  vector() {}

  // constractor ///////////////////////////////////////////////////////
  /**
   * @brief allocate size N vector memory space
   * @param[in] N vector length
   **/
  vector(const size_t N) { val.resize(N); }

  /**
   * @brief initialize size N vector, value to fill the container
   * @param[in] N vector length
   * @param[in] value fill Float type value to all elements
   **/
  vector(const size_t N, const Float value) { val.resize(N, value); }

  /**
   * @brief copy std::vector
   * @param[in] vec input std::vector
   **/
  vector(const std::vector<Float> &vec) {
    val.resize(vec.size());
    std::copy(vec.begin(), vec.end(), val.begin());
  }

  /**
   * @brief copy monolish::vector
   * @param[in] vec input monolish::vector
   **/
  vector(const vector<Float> &vec);

  /**
   * @brief copy from pointer
   * @param[in] start start pointer
   * @param[in] end  end pointer
   **/
  vector(const Float *start, const Float *end) {
    size_t size = (end - start);
    val.resize(size);
    std::copy(start, end, val.begin());
  }

  /**
   * @brief create N length rand(min~max) vector
   * @param[in] N vector length
   * @param[in] min rand min
   * @param[in] max rand max
   **/
  vector(const size_t N, const Float min, const Float max) {
    val.resize(N);
    std::random_device random;
    std::mt19937 mt(random());
    std::uniform_real_distribution<> rand(min, max);

    for (size_t i = 0; i < val.size(); i++) {
      val[i] = rand(mt);
    }
  }

  // communication
  // ///////////////////////////////////////////////////////////////////////////
  /**
   * @brief send data to GPU
   **/
  void send() const;

  /**
   * @brief recv and free data from GPU
   **/
  void recv();

  /**
   * @brief recv data from GPU (w/o free)
   **/
  void nonfree_recv();

  /**
   * @brief free data on GPU
   **/
  void device_free() const;

  /**
   * @brief false; // true: sended, false: not send
   * @return true is sended.
   * **/
  bool get_device_mem_stat() const { return gpu_status; }

  /**
   * @brief; free gpu mem.
   * **/
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
   **/
  Float *data() { return val.data(); }

  /**
   * @brief returns a direct pointer to the vector
   * @return A const pointer to the first element
   **/
  const Float *data() const { return val.data(); }

  /**
   * @brief resize vector (only CPU)
   * @param[in] N vector length
   **/
  void resize(size_t N) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    val.resize(N);
  }

  /**
   * @brief Add a new element at the ent of the vector (only CPU)
   * @param[in] val new element
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
   **/
  auto begin() { return val.begin(); }

  /**
   * @brief returns a end iterator
   * @return end iterator
   **/
  auto end() { return val.end(); }

  /**
   * @brief get vector size N
   * @return vector size
   **/
  auto size() const { return val.size(); }

  /**
   * @brief vector copy ( Copy the memory on CPU and GPU )
   * @return copied vector
   **/
  vector copy();

  /**
   * @brief print all elements to standart I/O
   **/
  void print_all() const {
    for (const auto v : val) {
      std::cout << v << std::endl;
    }
  }

  /**
   * @brief print all elements to file
   * @param[in] filename output filename
   **/
  void print_all(std::string filename) const {

    std::ofstream ofs(filename);
    if (!ofs) {
      throw std::runtime_error("error file cant open");
    }
    for (const auto v : val) {
      ofs << v << std::endl;
    }
  }

  // operator
  // ///////////////////////////////////////////////////////////////////////////

  /**
   * @brief copy vector, It is same as copy ( Copy the memory on CPU and GPU )
   * @param[in] vec source vector
   * @return output vector
   **/
  void operator=(const vector<Float> &vec);

  /**
   * @brief copy vector from std::vector (dont gpu copy)
   * @param[in] vec source std::vector
   * @return output vector
   **/
  void operator=(const std::vector<Float> &vec);

  // vec - scalar
  vector<Float> operator+(const Float value);
  void operator+=(const Float value);

  vector<Float> operator-(const Float value);
  void operator-=(const Float value);

  vector<Float> operator*(const Float value);
  void operator*=(const Float value);

  vector<Float> operator/(const Float value);
  void operator/=(const Float value);

  // vec - vec
  vector<Float> operator+(const vector<Float> &vec);
  void operator+=(const vector<Float> &vec);

  vector<Float> operator-(const vector<Float> &vec);
  void operator-=(const vector<Float> &vec);

  vector<Float> operator*(const vector<Float> &vec);
  void operator*=(const vector<Float> &vec);

  vector<Float> operator/(const vector<Float> &vec);
  void operator/=(const vector<Float> &vec);

  vector<Float> operator-();

  Float &operator[](size_t i) {
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    return val[i];
  }

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
};
} // namespace monolish
