#include "../../test_utils.hpp"

template <typename T> void ans_scal(double alpha, monolish::vector<T> &mx) {

  for (size_t i = 0; i < mx.size(); i++) {
    mx[i] = alpha * mx[i];
  }
}

template <typename T> bool test_send_scal(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);

  monolish::vector<T> ansx = x;
  ans_scal(alpha, ansx);

  x.send();
  monolish::blas::scal(alpha, x);
  x.recv();

  return ans_check<T>(__func__, x.data(), ansx.data(), x.size(), tol);
}

template <typename T> bool test_scal(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);

  monolish::vector<T> ansx = x;
  ans_scal(alpha, ansx);

  monolish::blas::scal(alpha, x);

  return ans_check<T>(__func__, x.data(), ansx.data(), x.size(), tol);
}
