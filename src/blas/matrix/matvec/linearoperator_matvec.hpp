#pragma once

namespace monolish {

namespace {
template <typename T, typename VEC1, typename VEC2>
void matvec_core(const matrix::LinearOperator<T> &A, const VEC1 &x, VEC2 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  if (A.get_matvec_init_flag()) {

    const size_t xoffset = x.get_offset();
    const size_t yoffset = y.get_offset();

    vector<T> x_tmp(x.size(), 0);
    if (x.get_device_mem_stat())
      x_tmp.send();
    vector<T> y_tmp(y.size(), 0);
    if (y.get_device_mem_stat())
      y_tmp.send();

    internal::vcopy(x.size(), x.data() + xoffset, x_tmp.data(),
                    x.get_device_mem_stat());

    y_tmp = A.get_matvec()(x_tmp);

    internal::vcopy(y.size(), y_tmp.data(), y.data() + yoffset,
                    y.get_device_mem_stat());
  } else {
    throw std::runtime_error("error matvec is not initialized");
  }

  logger.func_out();
}

template <typename T, typename VEC1, typename VEC2>
void rmatvec_core(const matrix::LinearOperator<T> &A, const VEC1 &x, VEC2 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == x.size());
  assert(A.get_col() == y.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  if (A.get_rmatvec_init_flag()) {
    const size_t xoffset = x.get_offset();
    const size_t yoffset = y.get_offset();

    vector<T> x_tmp(x.size(), 0);
    if (x.get_device_mem_stat())
      x_tmp.send();
    vector<T> y_tmp(y.size(), 0);
    if (y.get_device_mem_stat())
      y_tmp.send();

    internal::vcopy(x.size(), x.data() + xoffset, x_tmp.data(),
                    x.get_device_mem_stat());

    y_tmp = A.get_rmatvec()(x_tmp);

    internal::vcopy(y.size(), y_tmp.data(), y.data() + yoffset,
                    y.get_device_mem_stat());
  } else {
    throw std::runtime_error("error rmatmul is not initialized");
  }

  logger.func_out();
}

} // namespace

} // namespace monolish
