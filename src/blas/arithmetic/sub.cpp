#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

namespace monolish {
/////////////////////////////////////////////////
// vec - scalar
/////////////////////////////////////////////////
template <typename T> vector<T> vector<T>::operator-(const T value) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  vector<T> ans(val.size());

  T *vald = val.data();
  T *ansd = ans.data();
  size_t size = val.size();

  if(gpu_status==true){
#if MONOLISH_USE_GPU
      ans.send();
#pragma omp target teams distribute parallel for
      for (size_t i = 0; i < size; i++) {
          ansd[i] = vald[i] - value;
      }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  }
  else{
#pragma omp parallel for
      for (size_t i = 0; i < size; i++) {
          ansd[i] = vald[i] - value;
      }
  }

  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator-(const double value);
template vector<float> vector<float>::operator-(const float value);

template <typename T> void vector<T>::operator-=(const T value) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  vector<T> ans(val.size());

  T *vald = val.data();
  size_t size = val.size();

  if(gpu_status==true){
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
      for (size_t i = 0; i < size; i++) {
          vald[i] -= value;
      }
#else
      throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  }
  else{
#pragma omp parallel for
      for (size_t i = 0; i < size; i++) {
          vald[i] -= value;
      }
  }

  logger.func_out();
}
template void vector<double>::operator-=(const double value);
template void vector<float>::operator-=(const float value);

/////////////////////////////////////////////////
// vec - vec
/////////////////////////////////////////////////

template <typename T> vector<T> vector<T>::operator-(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (val.size() != vec.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  vector<T> ans(vec.size());

  const T *vecd = vec.data();
  T *vald = val.data();
  T *ansd = ans.data();
  size_t size = vec.size();

  if(gpu_status==true){
#if MONOLISH_USE_GPU
      ans.send();
#pragma omp target teams distribute parallel for
      for (size_t i = 0; i < size; i++) {
          ansd[i] = vald[i] - vecd[i];
      }
#else
      throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  }
  else{
#pragma omp parallel for
      for (size_t i = 0; i < size; i++) {
          ansd[i] = vald[i] - vecd[i];
      }
  }

  logger.func_out();
  return ans;
}

template vector<double> vector<double>::operator-(const vector<double> &vec);
template vector<float> vector<float>::operator-(const vector<float> &vec);

template <typename T> void vector<T>::operator-=(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (val.size() != vec.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  const T *vecd = vec.data();
  T *vald = val.data();
  size_t size = vec.size();

  if(gpu_status==true){
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
      for (size_t i = 0; i < size; i++) {
          vald[i] -= vecd[i];
      }
#else
      throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  }
  else{
#pragma omp parallel for
      for (size_t i = 0; i < size; i++) {
          vald[i] -= vecd[i];
      }
  }

  logger.func_out();
}
template void vector<double>::operator-=(const vector<double> &vec);
template void vector<float>::operator-=(const vector<float> &vec);
} // namespace monolish
