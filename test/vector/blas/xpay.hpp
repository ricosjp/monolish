#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T>
void ans_xpay(double alpha, monolish::vector<T> &mx, monolish::vector<T> &my) {
  if (mx.size() != my.size()) {
    throw std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    my[i] = mx[i] + alpha * my[i];
  }
}

template <typename T> bool test_send_xpay(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  monolish::vector<T> ansy = y;
  ans_xpay(alpha, x, ansy);

  monolish::util::send(x, y);
  monolish::blas::xpay(alpha, x, y);
  y.recv();

  return ans_check<T>(__func__, y.data(), ansy.data(), y.size(), tol);
}

template <typename T> bool test_xpay(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  monolish::vector<T> ansy = y;
  ans_xpay(alpha, x, ansy);

  monolish::blas::xpay(alpha, x, y);

  return ans_check<T>(__func__, y.data(), ansy.data(), y.size(), tol);
}
