#include "../test_utils.hpp"
#include "monolish_mpi.hpp"

template <typename T>
T ans_dot(monolish::vector<T> &mx, monolish::vector<T> &my) {
  if (mx.size() != my.size()) {
    throw std::runtime_error("x.size != y.size");
  }
  T ans = 0;

  for (size_t i = 0; i < mx.size(); i++) {
    ans += mx[i] * my[i];
  }

  return ans;
}

template <typename T> bool test_send_dot(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  T ans = ans_dot(x, y);

  monolish::util::send(x, y);
  T result = monolish::blas::dot(x, y);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename T> bool test_dot(const size_t size, double tol) {

  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  T ans = ans_dot(x, y);

  monolish::util::send(x, y);
  T result = monolish::blas::dot(x, y);

  return ans_check<T>(__func__, result, ans, tol);
}

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "error!, $1:vector size" << std::endl;
    return 1;
  }

  monolish::mpi::Comm &comm = monolish::mpi::Comm::get_instance();
  comm.Init();

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t size = atoi(argv[1]);
  std::cout << "size: " << size << std::endl;

  // nosend //
  if (test_dot<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_dot<float>(size, 1.0e-4) == false) {
    return 1;
  }

  // send //
  if (test_dot<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_dot<float>(size, 1.0e-4) == false) {
    return 1;
  }

  comm.Finalize();

  return 0;
}
