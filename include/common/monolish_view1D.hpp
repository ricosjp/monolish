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

namespace monolish {
  template <typename Float> class vector;

  namespace matrix {
    template <typename Float> class Dense;
    template <typename Float> class CRS;
    template <typename Float> class LinearOperator;
  }

/**
 * @brief 1D view class
 * @note
 * - Multi-threading: true
 * - GPU acceleration: true
 */
template <typename TYPE>
  class view1D {
    private:

      TYPE& target;
      size_t first;
      size_t last;
      size_t size;
      mutable bool gpu_status = false;

    public:
      view1D(TYPE& x, size_t start, size_t end):target(x){
        first = start;
        last = end;
        size = last - first; 
      }

      size_t get_size(){ return size;}

      size_t get_device_mem_stat(){ return target.get_device_mem_stat();}

      auto* data();

      void print_all();

      auto& operator[](size_t i);
  };

template<> auto* view1D<monolish::vector<double>>::data(){
  return target.data()+first;
}

template<> auto* view1D<monolish::matrix::Dense<double>>::data(){
  return target.val.data()+first;
}

template<> void view1D<monolish::vector<double>>::print_all(){
  for(size_t i=first; i<last; i++){
    std::cout << target[i] << std::endl;
  }
}

template<> void view1D<monolish::matrix::Dense<double>>::print_all(){
  for(size_t i=first; i<last; i++){
    std::cout << target.val[i] << std::endl;
  }
}

template<> auto& view1D<monolish::vector<double>>::operator[](size_t i){
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU vector cant use operator[]");
  }
  return target[first+i];
}

template<> auto& view1D<monolish::matrix::Dense<double>>::operator[](size_t i){
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU vector cant use operator[]");
  }
  return target.val[first+i];
}

} // namespace monolish
