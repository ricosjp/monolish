#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace monolish {
namespace matrix {

template <typename T> void COO<T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  std::cout << std::scientific;
  std::cout << std::setprecision(std::numeric_limits<T>::max_digits10);

  for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
    std::cout << row_index[i] + 1 << " " << col_index[i] + 1 << " " << vad[i]
              << std::endl;
  }

  logger.util_out();
}
template void COO<double>::print_all(bool force_cpu) const;
template void COO<float>::print_all(bool force_cpu) const;

template <typename T> void COO<T>::print_all(std::string filename) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  std::ofstream out(filename);
  out << std::scientific;
  out << std::setprecision(std::numeric_limits<T>::max_digits10);

  for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
    out << row_index[i] + 1 << " " << col_index[i] + 1 << " " << vad[i]
        << std::endl;
  }
  logger.util_out();
}
template void COO<double>::print_all(std::string filename) const;
template void COO<float>::print_all(std::string filename) const;

template <typename T> void COO<T>::output_mm(const std::string filename) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  std::ofstream out(filename);
  out << std::scientific;
  out << std::setprecision(std::numeric_limits<T>::max_digits10);

  out << (MM_BANNER " " MM_MAT " " MM_FMT " " MM_TYPE_REAL " " MM_TYPE_GENERAL)
      << std::endl;
  out << rowN << " " << colN << " " << get_nnz() << std::endl;

  for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
    out << row_index[i] + 1 << " " << col_index[i] + 1 << " " << vad[i]
        << std::endl;
  }
  logger.util_out();
}
template void COO<double>::output_mm(const std::string filename) const;
template void COO<float>::output_mm(const std::string filename) const;

template <typename T> void COO<T>::input_mm(const std::string filename) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  std::string banner, buf;
  std::string mm, mat, fmt, dtype, dstruct;

  // file open
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cerr << "Matrix.input: cannot open file " << filename << std::endl;
    std::exit(1);
  }

  // check Matrix Market bannner
  getline(ifs, banner);
  std::istringstream bn(banner);
  bn >> mm >> mat >> fmt >> dtype >> dstruct;

  if (mm != std::string(MM_BANNER)) {
    std::cerr << "Matrix.input: This matrix is not MM format:" << mm
              << std::endl;
    exit(-1);
  }
  if (mat != std::string(MM_MAT)) {
    std::cerr << "Matrix.input: This matrix is not matrix type:" << mat
              << std::endl;
    exit(-1);
  }
  if (fmt != std::string(MM_FMT)) {
    std::cerr << "Matrix.input: This matrix is not coodinate format:" << fmt
              << std::endl;
    exit(-1);
  }
  if (dtype != std::string(MM_TYPE_REAL)) {
    std::cerr << "Matrix.input: This matrix is not real:" << dtype << std::endl;
    exit(-1);
  }
  if (dstruct != std::string(MM_TYPE_GENERAL)) {
    std::cerr << "Matrix.input: This matrix is not general:" << dstruct
              << std::endl;
    exit(-1);
  }

  // skip %
  do {
    getline(ifs, buf);
  } while (buf[0] == '%');

  // check size
  size_t rowNN, colNN, NNZ;

  std::istringstream data(buf);
  data >> rowNN >> colNN >> NNZ;

  if (colNN <= 0 || NNZ < 0) {
    std::cerr << "Matrix.input: Matrix size should be positive" << std::endl;
    exit(-1);
  }

  rowN = rowNN;
  colN = colNN;

  // allocate
  row_index.resize(NNZ, 0.0);
  col_index.resize(NNZ, 0.0);
  vad_create_flag = true;
  resize(NNZ, 0.0);

  // set values
  for (auto i = decltype(NNZ){0}; i < NNZ; i++) {
    decltype(NNZ) ix, jx;
    T value;

    getline(ifs, buf);
    std::istringstream data(buf);
    data >> ix >> jx >> value;

    row_index[i] = ix - 1;
    col_index[i] = jx - 1;
    vad[i] = value;
  }
  logger.util_out();
}

template void COO<double>::input_mm(const std::string filename);
template void COO<float>::input_mm(const std::string filename);

} // namespace matrix
} // namespace monolish
