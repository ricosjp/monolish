#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {


template <typename T>
void matvec_core(const matrix::LinearOperator<T> &A, const vector<T> &x, vector<T> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_matvec_init_flag());

  y = A.get_matvec()(x);

  logger.func_out();
}

template <typename T>
void matvec_core(const matrix::LinearOperator<T> &A, const view1D<vector<T>, T>& x, vector<T>& y){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_matvec_init_flag());

  const size_t xoffset = x.get_offset();

  vector<T> x_tmp(x.size(), 0);
  if(x.get_device_mem_stat()) x_tmp.send();

  internal::vcopy(x.size(), x.data()+xoffset, x_tmp.data(), x.get_device_mem_stat());

  y = A.get_matvec()(x_tmp);

  logger.func_out();
}

template <typename T>
void matvec_core(const matrix::LinearOperator<T> &A, const vector<T>& x, view1D<vector<T>, T>& y){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_matvec_init_flag());

  const size_t yoffset = y.get_offset();

  vector<T> y_tmp(y.size(), 0);
  if(y.get_device_mem_stat()) y_tmp.send();

  y_tmp = A.get_matvec()(x);

  internal::vcopy(y.size(), y_tmp.data(), y.data()+yoffset, y.get_device_mem_stat());

  logger.func_out();
}


template <typename T>
void matvec_core(const matrix::LinearOperator<T> &A, const view1D<vector<T>, T>& x, view1D<vector<T>, T>& y){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_matvec_init_flag());

  const size_t xoffset = x.get_offset();
  const size_t yoffset = y.get_offset();

  vector<T> x_tmp(x.size(), 0);
  if(x.get_device_mem_stat()) x_tmp.send();
  vector<T> y_tmp(y.size(), 0);
  if(y.get_device_mem_stat()) y_tmp.send();

  internal::vcopy(x.size(), x.data()+xoffset, x_tmp.data(), x.get_device_mem_stat());

  y_tmp = A.get_matvec()(x_tmp);

  internal::vcopy(y.size(), y_tmp.data(), y.data()+yoffset, y.get_device_mem_stat());

  logger.func_out();
}

template <typename T>
void rmatvec_core(const matrix::LinearOperator<T> &A, const vector<T> &x, vector<T> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == x.size());
  assert(A.get_col() == y.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_rmatvec_init_flag());

  y = A.get_rmatvec()(x);

  logger.func_out();
}

template <typename T>
void rmatvec_core(const matrix::LinearOperator<T> &A, const vector<T> &x, view1D<vector<T>, T> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == x.size());
  assert(A.get_col() == y.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_rmatvec_init_flag());

  const size_t yoffset = y.get_offset();

  vector<T> y_tmp(y.size(), 0);
  if(y.get_device_mem_stat()) y_tmp.send();

  y_tmp = A.get_rmatvec()(x);

  internal::vcopy(y.size(), y_tmp.data(), y.data()+yoffset, y.get_device_mem_stat());

  logger.func_out();
}
template <typename T>
void rmatvec_core(const matrix::LinearOperator<T> &A, const view1D<vector<T>, T> &x, vector<T> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == x.size());
  assert(A.get_col() == y.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_rmatvec_init_flag());

  const size_t xoffset = x.get_offset();

  vector<T> x_tmp(x.size(), 0);
  if(x.get_device_mem_stat()) x_tmp.send();

  internal::vcopy(x.size(), x.data()+xoffset, x_tmp.data(), x.get_device_mem_stat());

  y = A.get_rmatvec()(x_tmp);

  logger.func_out();
}
template <typename T>
void rmatvec_core(const matrix::LinearOperator<T> &A, const view1D<vector<T>, T> &x, view1D<vector<T>, T> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == x.size());
  assert(A.get_col() == y.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_rmatvec_init_flag());

  const size_t xoffset = x.get_offset();
  const size_t yoffset = y.get_offset();

  vector<T> x_tmp(x.size(), 0);
  if(x.get_device_mem_stat()) x_tmp.send();
  vector<T> y_tmp(y.size(), 0);
  if(y.get_device_mem_stat()) y_tmp.send();

  internal::vcopy(x.size(), x.data()+xoffset, x_tmp.data(), x.get_device_mem_stat());

  y_tmp = A.get_rmatvec()(x_tmp);

  internal::vcopy(y.size(), y_tmp.data(), y.data()+yoffset, y.get_device_mem_stat());

  logger.func_out();
}

} // namespace

namespace blas {

void matvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
            vector<double> &y) {
  matvec_core(A, x, y);
}
void matvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
            view1D<vector<double>, double> &y) {
  matvec_core(A, x, y);
}
void matvec(const matrix::LinearOperator<double> &A, const view1D<vector<double>, double> &x,
            vector<double> &y) {
  matvec_core(A, x, y);
}
void matvec(const matrix::LinearOperator<double> &A, const view1D<vector<double>, double> &x,
            view1D<vector<double>, double> &y) {
  matvec_core(A, x, y);
}
void rmatvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
             vector<double> &y) {
  rmatvec_core(A, x, y);
}
void rmatvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
             view1D<vector<double>, double> &y) {
  rmatvec_core(A, x, y);
}
void rmatvec(const matrix::LinearOperator<double> &A, const view1D<vector<double>, double> &x,
             vector<double> &y) {
  rmatvec_core(A, x, y);
}
void rmatvec(const matrix::LinearOperator<double> &A, const view1D<vector<double>, double> &x,
             view1D<vector<double>, double> &y) {
  rmatvec_core(A, x, y);
}

void matvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
            vector<float> &y) {
  matvec_core(A, x, y);
}
void matvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
            view1D<vector<float>, float> &y) {
  matvec_core(A, x, y);
}
void matvec(const matrix::LinearOperator<float> &A, const view1D<vector<float>, float> &x,
            vector<float> &y) {
  matvec_core(A, x, y);
}
void matvec(const matrix::LinearOperator<float> &A, const view1D<vector<float>, float> &x,
            view1D<vector<float>, float> &y) {
  matvec_core(A, x, y);
}
void rmatvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
             vector<float> &y) {
  rmatvec_core(A, x, y);
}
void rmatvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
             view1D<vector<float>, float> &y) {
  rmatvec_core(A, x, y);
}
void rmatvec(const matrix::LinearOperator<float> &A, const view1D<vector<float>, float> &x,
             vector<float> &y) {
  rmatvec_core(A, x, y);
}
void rmatvec(const matrix::LinearOperator<float> &A, const view1D<vector<float>, float> &x,
             view1D<vector<float>, float> &y) {
  rmatvec_core(A, x, y);
}


} // namespace blas
} // namespace monolish
