#include "../benchmark_utils.hpp"

#define FUNC "sv_mul"
#define PERF 1 * size / time / 1.0e+9
#define MEM 2 * size * sizeof(T) / time / 1.0e+9

template <typename T> bool benchmark(const size_t size, const size_t iter) {

  // create random vector x rand(0.1~1.0)
  T value = 123.0;
  monolish::vector<T> x(size, 1.0);
  monolish::vector<T> ans(size, 321.0);

  monolish::util::send(x, ans);

  // exec
  auto start = std::chrono::system_clock::now();

  for (size_t i = 0; i < iter; i++) {
    monolish::vml::mul(x, value, ans);
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  // free device vector
  monolish::util::device_free(x, ans);

  double time = sec / iter;
  VECTOR_BLAS_OUTPUT_RESULT();

  return true;
}

int main(int argc, char **argv) {
  size_t size;
  size_t iter = VECTOR_BENCH_ITER;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");

  std::cout << VECTOR_BLAS_BENCH_HEADER << std::endl;

  // argv[1] is precision, argv[2] is size
  if (argc == 3) {
    size = atoi(argv[2]);

    if (strcmp(argv[1], "float") == 0) {
      benchmark<float>(size, iter);
    }

    if (strcmp(argv[1], "double") == 0) {
      benchmark<double>(size, iter);
    }

    return 0;
  }

  // default benchmark
  for (size = VECTOR_BENCH_MIN; size <= VECTOR_BENCH_MAX;
       size VECTOR_BENCH_INCL)
    benchmark<float>(size, iter);
  for (size = VECTOR_BENCH_MIN; size <= VECTOR_BENCH_MAX;
       size VECTOR_BENCH_INCL)
    benchmark<double>(size, iter);

  return 0;
}
