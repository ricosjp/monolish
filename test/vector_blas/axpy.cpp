#include "../test_utils.hpp"
#include "monolish_blas.hpp"

#define FUNC "axpy"
#define PERF 2 * size / time / 1.0e+9
#define MEM 3 * size * sizeof(T) / time / 1.0e+9

template <typename T>
void get_ans(double alpha, monolish::vector<T> &mx, monolish::vector<T> &my) {
  if (mx.size() != my.size()) {
    std::runtime_error("x.size != y.size");
  }

  for (size_t i = 0; i < mx.size(); i++) {
    my[i] = alpha * mx[i] + my[i];
  }
}

template <typename T>
bool test(const size_t size, double tol, const size_t iter,
          const size_t check_ans) {

  // create random vector x rand(0~1)
  double alpha = 123.0;
  monolish::vector<T> x(size, 0.0, 1.0);
  monolish::vector<T> y(size, 0.0, 1.0);

  monolish::vector<T> ansy = y;

  if (check_ans == 1) {
    get_ans(alpha, x, ansy);

    // send vector to device
    monolish::util::send(x, y);
    monolish::blas::axpy(alpha, x, y);

    // recv vector to device
    y.recv(); // recv. vector
    if (ans_check<T>(y.data(), ansy.data(), y.size(), tol) == false) {
      return false;
    }
    x.device_free();
  }

  monolish::util::send(x, y);

  // exec
  auto start = std::chrono::system_clock::now();

  for (size_t i = 0; i < iter; i++) {
    monolish::blas::axpy(alpha, x, y);
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  // free device vector
  monolish::util::device_free(x, y);

  double time = sec / iter;
  std::cout << "func\tprec\tsize\ttime[sec]\tperf[GFLOPS]\tmem[GB/s] "
            << std::endl;
  std::cout << FUNC << "\t" << std::flush;
  std::cout << get_type<T>() << "\t" << std::flush;
  std::cout << size << "\t" << std::flush;
  std::cout << time << "\t" << std::flush;
  std::cout << PERF << "\t" << std::flush;
  std::cout << MEM << "\t" << std::endl;

  return true;
}

int main(int argc, char **argv) {

  if (argc != 5) {
    std::cout << "error!, $1:precision (double or float), $2:vector size, $3: "
                 "iter, $4: error check (1/0)"
              << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t size = atoi(argv[2]);
  size_t iter = atoi(argv[3]);
  size_t check_ans = atoi(argv[4]);

  // exec and error check
  if (strcmp(argv[1], "double") == 0) {
    if (test<double>(size, 1.0e-8, iter, check_ans) == false) {
      std::cout << "error in double" << std::endl;
      return 1;
    }
  }

  if (strcmp(argv[1], "float") == 0) {
    if (test<float>(size, 1.0e-5, iter, check_ans) == false) {
      std::cout << "error in float" << std::endl;
      return 1;
    }
  }

  return 0;
}
