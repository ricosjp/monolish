#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
T tensor_COO<T>::at(const std::vector<size_t> &pos) const {

  // since last inserted element is effective elements,
  // checking from last element is necessary
  if (vad_nnz != 0) {
    for (auto k = vad_nnz; k > 0; --k) {
      if (index[k - 1] == pos) {
        return data()[k - 1];
      }
    }
  }
  return 0.0;
}
template double tensor_COO<double>::at(const std::vector<size_t> &pos) const;
template float tensor_COO<float>::at(const std::vector<size_t> &pos) const;

// insert //
template <typename T>
void tensor_COO<T>::insert(const std::vector<size_t> &pos, const T value) {

  if (vad_create_flag) {
    if (vad_nnz >= alloc_nnz) {
      size_t tmp = vad_nnz;
      alloc_nnz = 2 * alloc_nnz + 1;
      resize(alloc_nnz);
      vad_nnz = tmp;
    }
    index[vad_nnz] = pos;
    data()[vad_nnz] = value;
    vad_nnz++;
  } else {
    throw std::runtime_error("Error, not create coo matrix cant use insert");
  }
}
template void tensor_COO<double>::insert(const std::vector<size_t> &pos,
                                         const double value);
template void tensor_COO<float>::insert(const std::vector<size_t> &pos,
                                        const float value);

// sort //

template <typename T> void tensor_COO<T>::_q_sort(int lo, int hi) {
  if (lo >= hi) {
    return;
  }

  auto l = lo;
  auto h = hi;
  auto p = hi;
  auto p1 = index[p];
  double p3 = data()[p];
  int indl = (int)get_index(index[l]);
  int indh = (int)get_index(index[h]);
  int indp = (int)get_index(index[p]);

  do {
    while ((l < h) && ((indl - indp)) <= 0) {
      l = l + 1;
      indl = (int)get_index(index[l]);
    }
    while ((h > l) && ((indh - indp)) >= 0) {
      h = h - 1;
      indh = (int)get_index(index[h]);
    }
    if (l < h) {
      auto t = index[l];
      index[l] = index[h];
      index[h] = t;

      auto ti = indl;
      indl = indh;
      indh = ti;

      double td = data()[l];
      data()[l] = data()[h];
      data()[h] = td;
    }
  } while (l < h);

  index[p] = index[l];
  index[l] = p1;

  data()[p] = data()[l];
  data()[l] = p3;

  if (l - lo < hi - l) {
    _q_sort(lo, l - 1);
    _q_sort(l + 1, hi);
  } else {
    _q_sort(l + 1, hi);
    _q_sort(lo, l - 1);
  }
}
template void tensor_COO<double>::_q_sort(int lo, int hi);
template void tensor_COO<float>::_q_sort(int lo, int hi);

template <typename T> void tensor_COO<T>::sort(bool merge) {
  //  Sort by first Col and then Row
  //  TODO: This hand-written quick sort function should be retired
  //        after zip_iterator() (available in range-v3 library) is available in
  //        the standard (hopefully C++23)
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  _q_sort(0, vad_nnz - 1);

  /*  Remove duplicates */
  if (merge) {
    size_t k = 0;
    for (auto i = decltype(vad_nnz){1}; i < vad_nnz; i++) {
      if (index[k] != index[i]) {
        k++;
        index[k] = index[i];
      }
      data()[k] = data()[i];
    }
    vad_nnz = k + 1;
  }

  logger.util_out();
}
template void tensor_COO<double>::sort(bool merge);
template void tensor_COO<float>::sort(bool merge);

} // namespace tensor
} // namespace monolish
