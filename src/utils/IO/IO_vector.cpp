#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace monolish {

template <typename T> void vector<T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  const T *vald = val.data();

  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
    const size_t N = val.size();
#pragma omp target
    for (size_t i = 0; i < N; i++) {
      printf("%f\n", vald[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    for (size_t i = 0; i < val.size(); i++) {
      std::cout << vald[i] << std::endl;
    }
  }

  logger.util_out();
}
template void vector<double>::print_all(bool force_cpu) const;
template void vector<float>::print_all(bool force_cpu) const;

template <typename T> void vector<T>::print_all(std::string filename) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  std::ofstream ofs(filename);
  if (!ofs) {
    throw std::runtime_error("error file cant open");
  }
  for (const auto v : val) {
    ofs << v << std::endl;
  }
  logger.util_out();
}
template void vector<double>::print_all(std::string filename) const;
template void vector<float>::print_all(std::string filename) const;

} // namespace monolish
