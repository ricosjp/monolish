#include "../test_utils.hpp"
#include "monolish_blas.hpp"

#define FUNC "sum"
#define PERF 1 * size / time / 1.0e+9
#define MEM 1 * size * sizeof(T) / time / 1.0e+9

template <typename T> T get_ans(monolish::vector<T> &mx) {
  T ans = 0;

  for (size_t i = 0; i < mx.size(); i++) {
    ans += mx[i];
  }

  return ans;
}

template <typename T>
bool test(const size_t size, double tol, const size_t iter,
          const size_t check_ans) {

  // create random vector x rand(0~1)
  monolish::vector<T> x(size, 0.0, 1.0);

  // check ans
  if (check_ans == 1) {
    T ans = get_ans(x);

    x.send();
    T result = monolish::blas::sum(x);

    if (ans_check<T>(result, ans, tol) == false) {
      return false;
    }
    x.device_free();
  }

  x.send();
  // exec
  auto start = std::chrono::system_clock::now();

  for (size_t i = 0; i < iter; i++) {
    T result = monolish::blas::sum(x);
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

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
