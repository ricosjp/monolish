#include "../../include/common/monolish_dense.hpp"
#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

// todo: kill cerr

namespace monolish {
namespace matrix {

// matrix constructor

template <typename T>
COO<T>::COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
            const int *col, const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  nnz = NNZ;
  gpu_status = false;
  row_index.resize(nnz);
  col_index.resize(nnz);
  val.resize(nnz);

  std::copy(row, row + nnz, row_index.begin());
  std::copy(col, col + nnz, col_index.begin());
  std::copy(value, value + nnz, val.begin());
  logger.util_out();
}
template COO<double>::COO(const size_t M, const size_t N, const size_t NNZ,
                          const int *row, const int *col, const double *value);
template COO<float>::COO(const size_t M, const size_t N, const size_t NNZ,
                         const int *row, const int *col, const float *value);

template <typename T>
COO<T>::COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
            const int *col, const T *value, const size_t origin) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  nnz = NNZ;
  gpu_status = false;
  row_index.resize(nnz);
  col_index.resize(nnz);
  val.resize(nnz);

  std::copy(row, row + nnz, row_index.begin());
  std::copy(col, col + nnz, col_index.begin());
  std::copy(value, value + nnz, val.begin());

#pragma omp parallel for
  for (size_t i = 0; i < nnz; i++) {
    row_index[i] -= origin;
    col_index[i] -= origin;
  }
  logger.util_out();
}
template COO<double>::COO(const size_t M, const size_t N, const size_t NNZ,
                          const int *row, const int *col, const double *value,
                          const size_t origin);
template COO<float>::COO(const size_t M, const size_t N, const size_t NNZ,
                         const int *row, const int *col, const float *value,
                         const size_t origin);

template <typename T> COO<T>::COO(const matrix::COO<T> &coo) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = coo.get_row();
  colN = coo.get_col();
  nnz = coo.get_nnz();
  gpu_status = false;
  row_index.resize(nnz);
  col_index.resize(nnz);
  val.resize(nnz);
  std::copy(coo.row_index.data(), coo.row_index.data() + nnz,
            row_index.begin());
  std::copy(coo.col_index.data(), coo.col_index.data() + nnz,
            col_index.begin());
  std::copy(coo.val.data(), coo.val.data() + nnz, val.begin());
  logger.util_out();
}
template COO<double>::COO(const matrix::COO<double> &coo);
template COO<float>::COO(const matrix::COO<float> &coo);

// operator=
template <typename T> void COO<T>::operator=(const matrix::COO<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  if (get_row() != mat.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (get_col() != mat.get_col()) {
    throw std::runtime_error("error A.col != C.col");
  }
  if (get_nnz() != mat.get_nnz()) {
    throw std::runtime_error("error A.nnz != C.nnz");
  }
  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  // value copy
  internal::vcopy(get_nnz(), val.data(), mat.val.data(), get_device_mem_stat());

  logger.util_out();
}

// input and convert //////////////////////////////////////

template <typename T> void COO<T>::fill(T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < get_nnz(); i++) {
      val[i] = value;
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < get_nnz(); i++) {
      val[i] = value;
    }
  }
  logger.util_out();
}
template void COO<double>::fill(double value);
template void COO<float>::fill(float value);

template <typename T> void COO<T>::input_mm(const char *filename) {
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

  // symmetric check!
  if (colNN != rowNN) {
    std::cerr << "Matrix.input: Matrix is not square" << std::endl;
    exit(-1);
  }
  if (colNN <= 0 || NNZ < 0) {
    std::cerr << "Matrix.input: Matrix size should be positive" << std::endl;
    exit(-1);
  }

  rowN = rowNN;
  colN = rowN;
  nnz = NNZ;

  // allocate
  row_index.resize(nnz, 0.0);
  col_index.resize(nnz, 0.0);
  val.resize(nnz, 0.0);

  // set values
  for (size_t i = 0; i < nnz; i++) {
    size_t ix, jx;
    T value;

    getline(ifs, buf);
    std::istringstream data(buf);
    data >> ix >> jx >> value;

    row_index[i] = ix - 1;
    col_index[i] = jx - 1;
    val[i] = value;
  }
  logger.util_out();
}

template void COO<double>::input_mm(const char *filename);
template void COO<float>::input_mm(const char *filename);

template <typename T> void COO<T>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  std::cout << std::scientific;
  std::cout << std::setprecision(std::numeric_limits<T>::max_digits10);

  std::cout << (MM_BANNER " " MM_MAT " " MM_FMT " " MM_TYPE_REAL
                          " " MM_TYPE_GENERAL)
            << std::endl;
  std::cout << rowN << " " << colN << " " << nnz << std::endl;

  for (size_t i = 0; i < nnz; i++) {
    std::cout << row_index[i] + 1 << " " << col_index[i] + 1 << " " << val[i]
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

  out << (MM_BANNER " " MM_MAT " " MM_FMT " " MM_TYPE_REAL " " MM_TYPE_GENERAL)
      << std::endl;
  out << rowN << " " << colN << " " << nnz << std::endl;

  for (size_t i = 0; i < nnz; i++) {
    out << row_index[i] + 1 << " " << col_index[i] + 1 << " " << val[i]
        << std::endl;
  }
  logger.util_out();
}
template void COO<double>::print_all(std::string filename) const;
template void COO<float>::print_all(std::string filename) const;

template <typename T> T COO<T>::at(const size_t i, const size_t j) {
  if (i >= rowN || j >= colN) {
    throw std::out_of_range("error");
  }

  // since last inserted element is effective elements,
  // checking from last element is necessary
  if (nnz != 0) {
    for (size_t k = nnz; k > 0; --k) {
      if (row_index[k - 1] == (int)i && col_index[k - 1] == (int)j) {
        return val[k - 1];
      }
    }
  }
  return 0.0;
}
template double COO<double>::at(const size_t i, const size_t j);
template float COO<float>::at(const size_t i, const size_t j);

template <typename T> T COO<T>::at(const size_t i, const size_t j) const {
  if (i >= rowN || j >= colN) {
    throw std::out_of_range("error");
  }

  // since last inserted element is effective elements,
  // checking from last element is necessary
  if (nnz != 0) {
    for (size_t k = nnz; k > 0; --k) {
      if (row_index[k - 1] == (int)i && col_index[k - 1] == (int)j) {
        return val[k - 1];
      }
    }
  }
  return 0.0;
}
template double COO<double>::at(const size_t i, const size_t j) const;
template float COO<float>::at(const size_t i, const size_t j) const;

template <typename T>
void COO<T>::set_ptr(const size_t rN, const size_t cN,
                     const std::vector<int> &r, const std::vector<int> &c,
                     const std::vector<T> &v) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  col_index = c;
  row_index = r;
  val = v;

  rowN = rN;
  colN = cN;
  nnz = r.size();
  logger.util_out();
}
template void COO<double>::set_ptr(const size_t rN, const size_t cN,
                                   const std::vector<int> &r,
                                   const std::vector<int> &c,
                                   const std::vector<double> &v);
template void COO<float>::set_ptr(const size_t rN, const size_t cN,
                                  const std::vector<int> &r,
                                  const std::vector<int> &c,
                                  const std::vector<float> &v);

template <typename T>
void COO<T>::insert(const size_t m, const size_t n, const T value) {
  size_t rownum = m;
  if (rownum >= get_row()) {
    throw std::out_of_range("row index out of range");
  }
  size_t colnum = n;
  if (colnum >= get_col()) {
    throw std::out_of_range("column index out of range");
  }
  row_index.push_back(rownum);
  col_index.push_back(colnum);
  val.push_back(value);
  ++nnz;
}
template void COO<double>::insert(const size_t m, const size_t n,
                                  const double value);
template void COO<float>::insert(const size_t m, const size_t n,
                                 const float value);

template <typename T> void COO<T>::_q_sort(int lo, int hi) {
  // Very primitive quick sort
  if (lo >= hi) {
    return;
  }

  int l = lo;
  int h = hi;
  int p = hi;
  int p1 = row_index[p];
  int p2 = col_index[p];
  double p3 = val[p];

  do {
    while ((l < h) && ((row_index[l] != row_index[p])
                           ? (row_index[l] - row_index[p])
                           : (col_index[l] - col_index[p])) <= 0) {
      l = l + 1;
    }
    while ((h > l) && ((row_index[h] != row_index[p])
                           ? (row_index[h] - row_index[p])
                           : (col_index[h] - col_index[p])) >= 0) {
      h = h - 1;
    }
    if (l < h) {
      int t = row_index[l];
      row_index[l] = row_index[h];
      row_index[h] = t;

      t = col_index[l];
      col_index[l] = col_index[h];
      col_index[h] = t;

      double td = val[l];
      val[l] = val[h];
      val[h] = td;
    }
  } while (l < h);

  row_index[p] = row_index[l];
  row_index[l] = p1;

  col_index[p] = col_index[l];
  col_index[l] = p2;

  val[p] = val[l];
  val[l] = p3;

  /* Sort smaller array first for less stack usage */
  if (l - lo < hi - l) {
    _q_sort(lo, l - 1);
    _q_sort(l + 1, hi);
  } else {
    _q_sort(l + 1, hi);
    _q_sort(lo, l - 1);
  }
}
template void COO<double>::_q_sort(int lo, int hi);
template void COO<float>::_q_sort(int lo, int hi);

template <typename T> void COO<T>::sort(bool merge) {
  //  Sort by first Col and then Row
  //  TODO: This hand-written quick sort function should be retired
  //        after zip_iterator() (available in range-v3 library) is available in
  //        the standard (hopefully C++23)
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  _q_sort(0, nnz - 1);

  /*  Remove duplicates */
  if (merge) {
    size_t k = 0;
    for (size_t i = 1; i < nnz; i++) {
      if ((row_index[k] != row_index[i]) || (col_index[k] != col_index[i])) {
        k++;
        row_index[k] = row_index[i];
        col_index[k] = col_index[i];
      }
      val[k] = val[i];
    }
    nnz = k + 1;
  }

  logger.util_out();
}
template void COO<double>::sort(bool merge);
template void COO<float>::sort(bool merge);

template <typename T> void COO<T>::convert(const CRS<T> &crs) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_row(crs.get_row());
  set_col(crs.get_col());
  set_nnz(crs.get_nnz());

  row_index.resize(nnz);
  col_index.resize(nnz);
  val.resize(nnz);

  for (size_t i = 0; i < get_row(); i++) {
    for (size_t j = (size_t)crs.row_ptr[i]; j < (size_t)crs.row_ptr[i + 1];
         j++) {
      row_index[j] = i;
      col_index[j] = crs.col_ind[j];
      val[j] = crs.val[j];
    }
  }

  logger.util_out();
}
template void COO<double>::convert(const CRS<double> &crs);
template void COO<float>::convert(const CRS<float> &crs);

template <typename T> void COO<T>::convert(const Dense<T> &dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_row(dense.get_row());
  set_col(dense.get_col());
  row_index.resize(0);
  col_index.resize(0);
  val.resize(0);
  size_t nz = 0;

  for (size_t i = 0; i < dense.get_row(); i++) {
    for (size_t j = 0; j < dense.get_col(); j++) {
      if (dense.at(i, j) != 0) {
        row_index.push_back(i);
        col_index.push_back(j);
        val.push_back(dense.at(i, j));
        nz++;
      }
    }
  }
  set_nnz(nz);

  logger.util_out();
}
template void COO<double>::convert(const Dense<double> &dense);
template void COO<float>::convert(const Dense<float> &dense);

/// transpose /////

template <typename T> COO<T> &COO<T>::transpose() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  using std::swap;
  swap(rowN, colN);
  swap(row_index, col_index);
  return *this;
  logger.util_out();
}
template COO<double> &COO<double>::transpose();
template COO<float> &COO<float>::transpose();

template <typename T> void COO<T>::transpose(COO &B) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  B.set_row(get_col());
  B.set_col(get_row());
  B.set_nnz(get_nnz());
  B.row_index = get_col_ind();
  B.col_index = get_row_ptr();
  B.val = get_val_ptr();
  logger.util_out();
}
template void COO<double>::transpose(COO &B) const;
template void COO<float>::transpose(COO &B) const;

template <typename T> bool COO<T>::operator==(const COO<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU COO cant use operator==");
  }

  if (get_row() != mat.get_row()) {
    return false;
  }
  if (get_col() != mat.get_col()) {
    return false;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    return false;
  }

  if (get_device_mem_stat() == true) {
    if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), true))) {
      return false;
    }
    if (!(internal::vequal(get_nnz(), col_index.data(), mat.col_index.data(),
                           true))) {
      return false;
    }
    if (!(internal::vequal(get_nnz(), row_index.data(), mat.row_index.data(),
                           true))) {
      return false;
    }
  }

  if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), false))) {
    return false;
  }
  if (!(internal::vequal(get_nnz(), col_index.data(), mat.col_index.data(),
                         false))) {
    return false;
  }
  if (!(internal::vequal(get_nnz(), row_index.data(), mat.row_index.data(),
                         false))) {
    return false;
  }

  logger.util_out();
  return true;
}
template bool COO<double>::operator==(const COO<double> &mat) const;
template bool COO<float>::operator==(const COO<float> &mat) const;

template <typename T> bool COO<T>::operator!=(const COO<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU COO cant use operator!=");
  }

  if (get_row() != mat.get_row()) {
    return true;
  }
  if (get_col() != mat.get_col()) {
    return true;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    return true;
  }

  if (get_device_mem_stat() == true) {
    if (internal::vequal(get_nnz(), val.data(), mat.val.data(), true)) {
      return false;
    }
    if (internal::vequal(get_nnz(), col_index.data(), mat.col_index.data(),
                         true)) {
      return false;
    }
    if (internal::vequal(get_nnz(), row_index.data(), mat.row_index.data(),
                         true)) {
      return false;
    }
  }

  if (internal::vequal(get_nnz(), val.data(), mat.val.data(), false)) {
    return false;
  }
  if (internal::vequal(get_nnz(), col_index.data(), mat.col_index.data(),
                       false)) {
    return false;
  }
  if (internal::vequal(get_nnz(), row_index.data(), mat.row_index.data(),
                       false)) {
    return false;
  }

  logger.util_out();
  return true;
}
template bool COO<double>::operator!=(const COO<double> &mat) const;
template bool COO<float>::operator!=(const COO<float> &mat) const;

} // namespace matrix
} // namespace monolish
