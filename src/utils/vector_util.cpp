#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

/////vector constructor//////
template <typename T> vector<T>::vector(const size_t N) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val.resize(N);
  logger.util_out();
}
template vector<double>::vector(const size_t N);
template vector<float>::vector(const size_t N);

template <typename T> vector<T>::vector(const size_t N, const T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val.resize(N, value);
  logger.util_out();
}
template vector<double>::vector(const size_t N, const double value);
template vector<float>::vector(const size_t N, const float value);

template <typename T> vector<T>::vector(const std::vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val.resize(vec.size());
  std::copy(vec.begin(), vec.end(), val.begin());
  logger.util_out();
}
template vector<double>::vector(const std::vector<double> &vec);
template vector<float>::vector(const std::vector<float> &vec);

template <typename T> vector<T>::vector(const T *start, const T *end) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  size_t size = (end - start);
  val.resize(size);
  std::copy(start, end, val.begin());
  logger.util_out();
}
template vector<double>::vector(const double *start, const double *end);
template vector<float>::vector(const float *start, const float *end);

template <typename T>
vector<T>::vector(const size_t N, const T min, const T max) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val.resize(N);
  std::random_device random;
  std::mt19937 mt(random());
  std::uniform_real_distribution<> rand(min, max);

#pragma omp parallel for
  for (size_t i = 0; i < val.size(); i++) {
    val[i] = rand(mt);
  }
  logger.util_out();
}
template vector<double>::vector(const size_t N, const double min,
                                const double max);
template vector<float>::vector(const size_t N, const float min,
                               const float max);

// vector utils//////////////////////

template <typename T> void vector<T>::print_all() const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  for (const auto v : val) {
    std::cout << v << std::endl;
  }
  logger.util_out();
}
template void vector<double>::print_all() const;
template void vector<float>::print_all() const;

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

/// vector operator ///

template <typename T> bool vector<T>::operator==(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU vector cant use operator==");
  }

  if (val.size() != vec.size()){
    return false;
  }

  if(get_device_mem_stat()!=vec.get_device_mem_stat()){
    return false;
  }

  if(get_device_mem_stat()==true){
    bool ret = internal::vequal(val.size(), val.data(), vec.data(), true);
    if(ret == false){
      return false;
    }
  }
    bool ret = internal::vequal(val.size(), val.data(), vec.data(), false);

  logger.util_out();
  return ret;
}
template bool vector<double>::operator==(const vector<double> &vec);
template bool vector<float>::operator==(const vector<float> &vec);

template <typename T> bool vector<T>::operator!=(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU vector cant use operator==");
  }

  if (val.size() != vec.size()){
    return true;
  }

  if(get_device_mem_stat()!=vec.get_device_mem_stat()){
    return true;
  }

  if(get_device_mem_stat()==true){
    bool ret = internal::vequal(val.size(), val.data(), vec.data(), true);
    if(ret == false){
      return true;
    }
  }
    bool ret = internal::vequal(val.size(), val.data(), vec.data(), false);

  logger.util_out();
  return !ret;
}
template bool vector<double>::operator!=(const vector<double> &vec);
template bool vector<float>::operator!=(const vector<float> &vec);

template <typename T>
T util::get_residual_l2(matrix::CRS<T> &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  vector<T> tmp(x.size());
  tmp.send();

  blas::matvec(A, x, tmp); // tmp=Ax
  tmp = b - tmp;
  logger.util_out();
  return blas::nrm2(tmp);
}
template double util::get_residual_l2(matrix::CRS<double> &A, vector<double> &x,
                                      vector<double> &b);
template float util::get_residual_l2(matrix::CRS<float> &A, vector<float> &x,
                                     vector<float> &b);
} // namespace monolish
