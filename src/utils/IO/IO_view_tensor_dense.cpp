#include "../../../include/monolish_blas.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace monolish {

template <typename U, typename T> void view_tensor_Dense<U, T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
    const T *vald = target.data();
    auto shape = get_shape();
    int size = shape.size();
    const auto *shape_ptr = shape.data();
    auto ind = shape;
    auto ind_ptr = ind.data();
#pragma omp target
    for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
      auto i_copy = i;
      for (int j = size - 1; j >= 0; --j) {
        ind_ptr[j] = i_copy % shape_ptr[j];
        i_copy /= shape_ptr[j];
      }
      for (auto j = decltype(size){0}; j < size; j++) {
        printf("%lu ", ind_ptr[j]+1);
      }
      printf("%f\n", vald[first + i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  }else{
    for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
      auto ind = get_index(i);
      for (auto j = decltype(ind.size()){0}; j < ind.size(); j++) {
        std::cout << ind[j] + 1 << " ";
      }
      std::cout << data()[first + i] << std::endl;
    }
  }

  logger.util_out();
}
template void view_tensor_Dense<vector<double>, double>::print_all(bool force_cpu) const;
template void view_tensor_Dense<matrix::Dense<double>, double>::print_all(bool force_cpu) const;
//template void view_tensor_Dense<matrix::LinearOperator<double>, double>::print_all(bool force_cpu) const;
template void view_tensor_Dense<tensor::tensor_Dense<double>, double>::print_all(bool force_cpu) const;
template void view_tensor_Dense<vector<float>, float>::print_all(bool force_cpu) const;
template void view_tensor_Dense<matrix::Dense<float>, float>::print_all(bool force_cpu) const;
//template void view_tensor_Dense<matrix::LinearOperator<float>, float>::print_all(bool force_cpu) const;
template void view_tensor_Dense<tensor::tensor_Dense<float>, float>::print_all(bool force_cpu) const;


} // namespace monolish
