#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> T COO<T>::at(const size_t i, const size_t j) const {

  assert(i <= get_row());
  assert(j <= get_col());

  // since last inserted element is effective elements,
  // checking from last element is necessary
  if (vad_nnz != 0) {
    for (auto k = vad_nnz; k > 0; --k) {
      if (row_index[k - 1] == (int)i && col_index[k - 1] == (int)j) {
        return data()[k - 1];
      }
    }
  }
  return 0.0;
}
template double COO<double>::at(const size_t i, const size_t j) const;
template float COO<float>::at(const size_t i, const size_t j) const;

// insert //
template <typename T>
void COO<T>::insert(const size_t m, const size_t n, const T value) {

  if (vad_create_flag) {
    auto rownum = m;
    auto colnum = n;
    assert(rownum <= get_row());
    assert(colnum <= get_col());

    if (vad_nnz >= alloc_nnz) {
      size_t tmp = vad_nnz;
      alloc_nnz = 2 * alloc_nnz + 1;
      resize(alloc_nnz);
      vad_nnz = tmp;
    }
    row_index[vad_nnz] = rownum;
    col_index[vad_nnz] = colnum;
    data()[vad_nnz] = value;
    vad_nnz++;
  } else {
    throw std::runtime_error("Error, not create coo matrix cant use insert");
  }
}
template void COO<double>::insert(const size_t m, const size_t n,
                                  const double value);
template void COO<float>::insert(const size_t m, const size_t n,
                                 const float value);

// sort //

template <typename T> void COO<T>::_q_sort(int lo, int hi) {
  // Very primitive quick sort
  if (lo >= hi) {
    return;
  }

  auto l = lo;
  auto h = hi;
  auto p = hi;
  auto p1 = row_index[p];
  auto p2 = col_index[p];
  double p3 = data()[p];

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
      auto t = row_index[l];
      row_index[l] = row_index[h];
      row_index[h] = t;

      t = col_index[l];
      col_index[l] = col_index[h];
      col_index[h] = t;

      double td = data()[l];
      data()[l] = data()[h];
      data()[h] = td;
    }
  } while (l < h);

  row_index[p] = row_index[l];
  row_index[l] = p1;

  col_index[p] = col_index[l];
  col_index[l] = p2;

  data()[p] = data()[l];
  data()[l] = p3;

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

  _q_sort(0, vad_nnz - 1);

  /*  Remove duplicates */
  if (merge) {
    size_t k = 0;
    for (auto i = decltype(vad_nnz){1}; i < vad_nnz; i++) {
      if ((row_index[k] != row_index[i]) || (col_index[k] != col_index[i])) {
        k++;
        row_index[k] = row_index[i];
        col_index[k] = col_index[i];
      }
      data()[k] = data()[i];
    }
    vad_nnz = k + 1;
  }

  logger.util_out();
}
template void COO<double>::sort(bool merge);
template void COO<float>::sort(bool merge);

} // namespace matrix
} // namespace monolish
