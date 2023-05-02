#include "../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T, typename V> void ans_axpy(T alpha, V &mx, V &my) {
  if (mx.size() != my.size()) {
    throw std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    my[i] = alpha * mx[i] + my[i];
  }
}

template <typename U, typename T>
bool test_send_axpy_core(U &vecx, U &vecy, double tol) {
  T alpha = 123.0;
  const size_t size = vecx.get_nnz();

  monolish::view1D<U, T> x(vecx, 5, size / 2);
  monolish::view1D<U, T> y(vecy, 5, size / 2);

  U vecansy = vecy;
  monolish::view1D<U, T> ansy(vecansy, 5, size / 2);
  ans_axpy(alpha, x, ansy);

  monolish::util::send(vecx, vecy);
  monolish::blas::axpy(alpha, x, y);
  vecy.recv();

  return ans_check<T>(__func__, vecy.data(), vecansy.data(), vecy.get_nnz(),
                      tol);
}

template <typename U, typename T>
bool test_axpy_core(U &vecx, U &vecy, double tol) {
  T alpha = 123.0;
  const size_t size = vecx.get_nnz();

  monolish::view1D<U, T> x(vecx, 5, size / 2);
  monolish::view1D<U, T> y(vecy, 5, size / 2);

  U vecansy = vecy;
  monolish::view1D<U, T> ansy(vecansy, 5, size / 2);
  ans_axpy(alpha, x, ansy);

  monolish::blas::axpy(alpha, x, y);

  return ans_check<T>(__func__, y.data(), ansy.data(), y.get_nnz(), tol);
}

template <typename U, typename T>
bool test_send_axpy(const size_t size, double tol) {
  U vecx(size, 0.0, 1.0);
  U vecy(size, 0.0, 1.0);
  return test_send_axpy_core<U, T>(vecx, vecy, tol);
}

template <typename U, typename T>
bool test_axpy(const size_t size, double tol) {
  U vecx(size, 0.0, 1.0);
  U vecy(size, 0.0, 1.0);
  return test_axpy_core<U, T>(vecx, vecy, tol);
}

template <typename U, typename T>
bool test_send_axpy(const size_t size1, const size_t size2, double tol) {
  U vecx(size1, size2, 0.0, 1.0);
  U vecy(size1, size2, 0.0, 1.0);
  return test_send_axpy_core<U, T>(vecx, vecy, tol);
}

template <typename U, typename T>
bool test_axpy(const size_t size1, const size_t size2, double tol) {
  U vecx(size1, size2, 0.0, 1.0);
  U vecy(size1, size2, 0.0, 1.0);
  return test_axpy_core<U, T>(vecx, vecy, tol);
}

template <typename U, typename T>
bool test_send_axpy(const size_t size1, const size_t size2, const size_t size3,
                    double tol) {
  U vecx({size1, size2, size3}, 0.0, 1.0);
  U vecy({size1, size2, size3}, 0.0, 1.0);
  return test_send_axpy_core<U, T>(vecx, vecy, tol);
}

template <typename U, typename T>
bool test_axpy(const size_t size1, const size_t size2, const size_t size3,
               double tol) {
  U vecx({size1, size2, size3}, 0.0, 1.0);
  U vecy({size1, size2, size3}, 0.0, 1.0);
  return test_axpy_core<U, T>(vecx, vecy, tol);
}

int main() {
  size_t size = 40;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (test_axpy<monolish::vector<double>, double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_axpy<monolish::vector<float>, float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_axpy<monolish::vector<double>, double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_axpy<monolish::vector<float>, float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_axpy<monolish::matrix::Dense<double>, double>(size, size, 1.0e-8) ==
      false) {
    return 1;
  }
  if (test_axpy<monolish::matrix::Dense<float>, float>(size, size, 1.0e-4) ==
      false) {
    return 1;
  }

  if (test_send_axpy<monolish::matrix::Dense<double>, double>(
          size, size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_axpy<monolish::matrix::Dense<float>, float>(size, size,
                                                            1.0e-4) == false) {
    return 1;
  }

  if (test_axpy<monolish::tensor::tensor_Dense<double>, double>(
          size, size, size, 1.0e-8) == false) {
    return 1;
  }
  if (test_axpy<monolish::tensor::tensor_Dense<float>, float>(
          size, size, size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_axpy<monolish::tensor::tensor_Dense<double>, double>(
          size, size, size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_axpy<monolish::tensor::tensor_Dense<float>, float>(
          size, size, size, 1.0e-4) == false) {
    return 1;
  }
}
