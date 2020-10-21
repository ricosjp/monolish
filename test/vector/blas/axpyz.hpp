#include "../../test_utils.hpp"

template <typename T>
void ans_axpyz(double alpha, monolish::vector<T> &mx, monolish::vector<T> &my,
               monolish::vector<T> &mz) {
  if (mx.size() != my.size() || mx.size() != mz.size()) {
    throw std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    mz[i] = alpha * mx[i] + my[i];
  }
}

template <typename T> bool test_send_axpyz(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);
  monolish::vector<T> z(size, 0.0, 1.0);

  monolish::vector<T> ansz = z;
  ans_axpyz(alpha, x, y, ansz);

  monolish::util::send(x, y, z);
  monolish::blas::axpyz(alpha, x, y, z);
  z.recv();

  return ans_check<T>(__func__, z.data(), ansz.data(), z.size(), tol);
}

template <typename T> bool test_axpyz(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);
  monolish::vector<T> z(size, 0.0, 1.0);

  monolish::vector<T> ansz = z;
  ans_axpyz(alpha, x, y, ansz);

  monolish::blas::axpyz(alpha, x, y, z);

  return ans_check<T>(__func__, z.data(), ansz.data(), z.size(), tol);
}
