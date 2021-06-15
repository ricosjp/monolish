#include "../benchmark_utils.hpp"

#define FUNC "LU"
#define DENSE_PERF                                                             \
  2.0 / 3.0 *                                                                  \
      ((double)size / 1000 * (double)size / 1000 * (double)size / 1000) / time

template <typename MAT_A, typename T>
bool benchmark(const size_t size, const size_t iter) {

  MAT_A A(size, size, 0.0, 1.0);
  monolish::vector<T> b(size, 123.0);

  monolish::util::send(A, b);

  monolish::equation::LU<MAT_A, T> LU_solver;

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < iter; i++) {
    LU_solver.solve(A, b);
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  A.device_free();
  b.device_free();

  double time = sec / iter;
  std::cout << FUNC << "(" << A.type() << ")\t" << std::flush;
  std::cout << get_type<T>() << "\t" << std::flush;
  std::cout << size << "\t" << std::flush;
  std::cout << time << "\t" << std::flush;

  if ((strcmp(A.type().data(), "Dense") == 0)) {
    std::cout << DENSE_PERF << "\t" << std::endl;
  }

  return true;
}

int main(int argc, char **argv) {

  if (argc <= 1) {
    std::cout << "error $1: format of A (only Dense now)" << std::endl;
    return 1;
  }

  if ((strcmp(argv[1], "Dense") != 0)) {
    return 1;
  }

  std::cout << "func\tkind\tprec\tsize\ttime[sec]\tperf[GFLOPS]" << std::endl;

  size_t iter = LU_BENCH_ITER;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");

  // Dense
  for (size_t size = LU_NNN_BENCH_MIN; size <= LU_NNN_BENCH_MAX;
       size LU_NNN_BENCH_ITER) {
    benchmark<monolish::matrix::Dense<float>, float>(size, iter);
  }

  for (size_t size = LU_NNN_BENCH_MIN; size <= LU_NNN_BENCH_MAX;
       size LU_NNN_BENCH_ITER) {
    benchmark<monolish::matrix::Dense<double>, double>(size, iter);
  }

  return 0;
}
