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

template <typename U, typename T>
void view_Dense<U, T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
    const T *vald = target.data();
#pragma omp target
    for (auto i = decltype(get_row()){0}; i < get_row(); i++) {
      for (auto j = decltype(get_col()){0}; j < get_col(); j++) {
        printf("%lu %lu %f\n", i + 1, j + 1, vald[first + i * get_col() + j]);
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    for (auto i = decltype(get_row()){0}; i < get_row(); i++) {
      for (auto j = decltype(get_col()){0}; j < get_col(); j++) {
        std::cout << i + 1 << " " << j + 1 << " "
                  << data()[first + i * get_col() + j] << std::endl;
      }
    }
  }

  logger.util_out();
}
template void
view_Dense<vector<double>, double>::print_all(bool force_cpu) const;
template void
view_Dense<matrix::Dense<double>, double>::print_all(bool force_cpu) const;
// template void view_Dense<matrix::LinearOperator<double>,
// double>::print_all(bool force_cpu) const;
template void view_Dense<tensor::tensor_Dense<double>, double>::print_all(
    bool force_cpu) const;
template void view_Dense<vector<float>, float>::print_all(bool force_cpu) const;
template void
view_Dense<matrix::Dense<float>, float>::print_all(bool force_cpu) const;
// template void view_Dense<matrix::LinearOperator<float>,
// float>::print_all(bool force_cpu) const;
template void
view_Dense<tensor::tensor_Dense<float>, float>::print_all(bool force_cpu) const;

} // namespace monolish
