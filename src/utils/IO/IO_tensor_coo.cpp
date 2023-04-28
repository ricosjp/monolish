#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace monolish {
namespace tensor {

template <typename T> void tensor_COO<T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  std::cout << std::scientific;
  std::cout << std::setprecision(std::numeric_limits<T>::max_digits10);

  for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
    for (auto j = decltype(index[i].size()){0}; j < index[i].size(); j++) {
      std::cout << index[i][j] + 1 << " ";
    }
    std::cout << data()[i] << std::endl;
  }

  logger.util_out();
}
template void tensor_COO<double>::print_all(bool force_cpu) const;
template void tensor_COO<float>::print_all(bool force_cpu) const;

template <typename T>
void tensor_COO<T>::print_all(std::string filename) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  std::ofstream out(filename);
  out << std::scientific;
  out << std::setprecision(std::numeric_limits<T>::max_digits10);

  for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
    for (auto j = decltype(index[i].size()){0}; j < index[i].size(); j++) {
      out << index[i][j] + 1 << " ";
    }
    out << data()[i] << std::endl;
  }
  logger.util_out();
}
template void tensor_COO<double>::print_all(std::string filename) const;
template void tensor_COO<float>::print_all(std::string filename) const;

} // namespace tensor
} // namespace monolish
