#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace{
  template <typename V1, typename V2>
    bool equal_core(const V1 &vec1, const V2 &vec2, bool compare_cpu_and_device){
      Logger &logger = Logger::get_instance();
      logger.util_in(monolish_func);

      if (vec1.size() != vec2.size()) {
        return false;
      }
      if (vec1.get_device_mem_stat() != vec2.get_device_mem_stat()) {
        return false;
      }

      if (vec1.get_device_mem_stat() == true) {
        if (!(internal::vequal(vec1.size(), vec1.data()+vec1.get_offset(), vec2.data()+vec2.get_offset(), true))) {
          return false;
        }
      } else if (vec1.get_device_mem_stat() == false ||
          compare_cpu_and_device == false) {
        if (!(internal::vequal(vec1.size(), vec1.data()+vec1.get_offset(), vec2.data()+vec2.get_offset(), false))) {
          return false;
        }
      }

      logger.util_out();
      return true;
    }
}

template<typename T>
bool vector<T>::equal(const vector<T> &vec, bool compare_cpu_and_device) const{
  return equal_core(*this, vec, compare_cpu_and_device);
}
template bool vector<double>::equal(const vector<double> &vec, bool compare_cpu_and_device) const;
template bool vector<float>::equal(const vector<float> &vec, bool compare_cpu_and_device) const;

template<typename T>
bool vector<T>::equal(const view1D<vector<T>,T> &vec, bool compare_cpu_and_device) const{
  return equal_core(*this, vec, compare_cpu_and_device);
}
template bool vector<double>::equal(const view1D<vector<double>,double> &vec, bool compare_cpu_and_device) const;
template bool vector<float>::equal(const view1D<vector<float>,float> &vec, bool compare_cpu_and_device) const;

template<typename T>
bool vector<T>::equal(const view1D<matrix::Dense<T>,T> &vec, bool compare_cpu_and_device) const{
  return equal_core(*this, vec, compare_cpu_and_device);
}
template bool vector<double>::equal(const view1D<matrix::Dense<double>,double> &vec, bool compare_cpu_and_device) const;
template bool vector<float>::equal(const view1D<matrix::Dense<float>,float> &vec, bool compare_cpu_and_device) const;


template<typename T>
bool vector<T>::operator==(const vector<T> &vec) const{
  return equal_core(*this, vec, false);
}
template bool vector<double>::operator==(const vector<double> &vec) const;
template bool vector<float>::operator==(const vector<float> &vec) const;

template<typename T>
bool vector<T>::operator==(const view1D<vector<T>,T> &vec) const{
  return equal_core(*this, vec, false);
}
template bool vector<double>::operator==(const view1D<vector<double>,double> &vec) const;
template bool vector<float>::operator==(const view1D<vector<float>,float> &vec) const;

template<typename T>
bool vector<T>::operator==(const view1D<matrix::Dense<T>,T> &vec) const
{
  return equal_core(*this, vec, false);
}
template bool vector<double>::operator==(const view1D<matrix::Dense<double>,double> &vec) const;
template bool vector<float>::operator==(const view1D<matrix::Dense<float>,float> &vec) const;


template<typename T>
bool vector<T>::operator!=(const vector<T> &vec) const{
  return !equal_core(*this, vec, false);
}
template bool vector<double>::operator!=(const vector<double> &vec) const;
template bool vector<float>::operator!=(const vector<float> &vec) const;

template<typename T>
bool vector<T>::operator!=(const view1D<vector<T>,T> &vec) const{
  return !equal_core(*this, vec, false);
}
template bool vector<double>::operator!=(const view1D<vector<double>,double> &vec) const;
template bool vector<float>::operator!=(const view1D<vector<float>,float> &vec) const;

template<typename T>
bool vector<T>::operator!=(const view1D<matrix::Dense<T>,T> &vec) const
{
  return !equal_core(*this, vec, false);
}
template bool vector<double>::operator!=(const view1D<matrix::Dense<double>,double> &vec) const;
template bool vector<float>::operator!=(const view1D<matrix::Dense<float>,float> &vec) const;

} // namespace monolish
