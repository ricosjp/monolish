#include "../../test_utils.hpp"

template <typename T>
void ans_vector_times(monolish::vector<T> &ma, monolish::vector<T> &mb,
                      monolish::vector<T> &my) {

  for (size_t i = 0; i < ma.size(); i++) {
    my[i] = ma[i] * mb[i];
  }
}

template <typename T>
bool test_send_vector_times(const size_t size, double tol) {

  monolish::vector<T> a(size, 0.0, 1.0, test_random_engine());
  monolish::vector<T> b(size, 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(size, 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy = y;
  ans_vector_times(a, b, ansy);

  a.send();
  b.send();
  y.send();
  monolish::blas::times(a, b, y);
  y.recv();

  return ans_check<T>(__func__, y.begin(), ansy.begin(), y.size(), tol);
}

template <typename T> bool test_vector_times(const size_t size, double tol) {

  monolish::vector<T> a(size, 0.0, 1.0, test_random_engine());
  monolish::vector<T> b(size, 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(size, 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy = y;
  ans_vector_times(a, b, ansy);

  monolish::blas::times(a, b, y);

  return ans_check<T>(__func__, y.begin(), ansy.begin(), y.size(), tol);
}
