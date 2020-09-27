#include "../test_utils.hpp"
#include "monolish_blas.hpp"

#define FUNC "tanh"
#define PERF size / time / 1.0e+9
#define MEM 2 * size * sizeof(T) / time / 1.0e+9

template <typename T>
void get_ans(monolish::vector<T> &ans) {
  for (size_t i = 0; i < ans.size(); i++) {
    ans[i] = std::tanh(ans[i]);
  }
}

template <typename T>
bool test(const size_t size, double tol, const size_t iter,const size_t check_ans) {

  // create random vector x rand(0.1~10.0)
  monolish::vector<T> ans(size, 0.1, 10.0);

  monolish::vector<T> ans_tmp;

  // copy
  ans_tmp = ans.copy();

  // check arithmetic
  if (check_ans == 1) {
    get_ans(ans_tmp);

    monolish::util::send(ans);
    ans.tanh();
    ans.recv();

    if (ans_check<T>(ans.data(), ans_tmp.data(), ans.size(), tol) == false) {
      return false;
    }
  }

  ans.send();

  // exec
  auto start = std::chrono::system_clock::now();

  for (size_t i = 0; i < iter; i++) {
    ans.tanh();
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  // free device vector
  monolish::util::device_free(ans);

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
      if (test<double>(size, 1.0e-6, iter,check_ans) == false) {
          return 1;
      }
  }

  if (strcmp(argv[1], "float") == 0) {
      if (test<float>(size, 1.0e-4, iter,check_ans) == false) {
          return 1;
      }
  }

  return 0;
}
