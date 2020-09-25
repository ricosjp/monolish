#include "../../include/common/monolish_dense.hpp"
#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../../include/common/monolish_vector.hpp"
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

// kill cerr

namespace monolish {
namespace matrix {

template <typename T> void Dense<T>::print_all() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU matrix cant use print_all");
  }

  for (size_t i = 0; i < get_row(); i++) {
    for (size_t j = 0; j < get_col(); j++) {
      std::cout << i + 1 << " " << j + 1 << " " << val[i * get_col() + j]
                << std::endl;
    }
  }

  logger.util_out();
}
template void Dense<double>::print_all();
template void Dense<float>::print_all();

template <typename T> void Dense<T>::convert(const COO<T> &coo) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_row(coo.get_row());
  set_col(coo.get_col());
  set_nnz(get_row() * get_col());
  val.resize(get_row() * get_col());

#pragma omp parallel for
  for (size_t i = 0; i < get_nnz(); i++) {
    val[i] = 0.0;
  }

  for (size_t i = 0; i < coo.get_nnz(); i++) {
    insert(coo.row_index[i], coo.col_index[i], coo.val[i]);
  }
  logger.util_out();
}
template void Dense<double>::convert(const COO<double> &coo);
template void Dense<float>::convert(const COO<float> &coo);
} // namespace matrix
} // namespace monolish
