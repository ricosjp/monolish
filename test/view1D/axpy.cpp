#include "../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T, typename V>
void ans_axpy(T alpha, V &mx, V &my) {
  if (mx.size() != my.size()) {
    throw std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    my[i] = alpha * mx[i] + my[i];
  }
}

template <typename T> bool test_send_axpy(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> vecx(size, 1.0);
  monolish::vector<T> vecy(size, 1.0);
  monolish::view1D<monolish::vector<T>,T> x(vecx, 5, size/2);
  monolish::view1D<monolish::vector<T>,T> y(vecy, 5, size/2);

  monolish::vector<T> vecansy = vecy;
  monolish::view1D<monolish::vector<T>,T> ansy(vecansy, 5, size/2);
  ans_axpy(alpha, x, ansy);

  monolish::util::send(vecx, vecy);
  monolish::blas::axpy(alpha, x, y);
  vecy.recv();

  return ans_check<T>(__func__, vecy.data(), vecansy.data(), vecy.size(), tol);
}

template <typename T> bool test_axpy(const size_t size, double tol) {

  T alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  monolish::vector<T> ansy = y;
  ans_axpy(alpha, x, ansy);

  monolish::blas::axpy(alpha, x, y);

  return ans_check<T>(__func__, y.data(), ansy.data(), y.size(), tol);
}

int main(){
  size_t size = 40;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (test_axpy<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_axpy<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_axpy<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_axpy<float>(size, 1.0e-4) == false) {
    return 1;
  }
}
