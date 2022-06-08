#include "../../test_utils.hpp"

template <typename T>
void ans_scalar_times(double alpha, monolish::vector<T> &mx,
                      monolish::vector<T> &my) {

  for (size_t i = 0; i < mx.size(); i++) {
    my[i] = alpha * mx[i];
  }
}

template <typename T>
bool test_send_scalar_times(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(size, 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy = y;
  ans_scalar_times(alpha, x, ansy);

  x.send();
  y.send();
  monolish::blas::times(alpha, x, y);
  y.recv();

  return ans_check<T>(__func__, y.data(), ansy.data(), y.size(), tol);
}

template <typename T> bool test_scalar_times(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(size, 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy = y;
  ans_scalar_times(alpha, x, ansy);

  monolish::blas::times(alpha, x, y);

  return ans_check<T>(__func__, y.data(), ansy.data(), y.size(), tol);
}
