#include "../benchmark_utils.hpp"

#define FUNC "transpose"

template <typename MAT, typename T>
bool benchmark(const size_t M, const size_t N, int iter) {

  size_t nnzrow = 81;
  if (nnzrow < N) {
    nnzrow = 81;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::band_matrix<T>(M, N, nnzrow, 1.0, 1.0);

  MAT A(seedA); // M*N matrix

  if (A.type() != "COO") {
    monolish::util::send(A);
  }

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < iter; i++) {
    A.transpose();
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  A.device_free();

  double time = sec / iter;
  std::cout << FUNC << "(" << A.type() << ")\t" << std::flush;
  std::cout << A.type() << "\t" << std::flush;
  std::cout << get_type<T>() << "\t" << std::flush;
  std::cout << M << "\t" << std::flush;
  std::cout << N << "\t" << std::flush;
  std::cout << time << std::endl;

  return true;
}

int main(int argc, char **argv) {

  if (argc <= 1) {
    std::cout << "$1: format" << std::endl;
    return 1;
  }

  std::cout << "func\tkind\tprec\tM\tN\ttime[sec]" << std::endl;
  size_t iter = MATRIX_BENCH_ITER;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (argc == 5) {
    const size_t M = atoi(argv[3]);
    const size_t N = atoi(argv[4]);
    if (strcmp(argv[2], "double") == 0) {
      if ((strcmp(argv[1], "Dense") == 0)) {
        benchmark<monolish::matrix::Dense<double>, double>(M, N, iter);
      }
      if ((strcmp(argv[1], "COO") == 0)) {
        benchmark<monolish::matrix::COO<double>, double>(M, N, iter);
      }
    }

    if (strcmp(argv[2], "float") == 0) {
      if ((strcmp(argv[1], "Dense") == 0)) {
        benchmark<monolish::matrix::Dense<float>, float>(M, N, iter);
      }
      if ((strcmp(argv[1], "COO") == 0)) {
        benchmark<monolish::matrix::COO<float>, float>(M, N, iter);
      }
    }
  }

  // Dense
  if ((strcmp(argv[1], "Dense") == 0)) {
    for (size_t size = DENSE_NN_BENCH_MIN; size <= DENSE_NN_BENCH_MAX;
         size += 1000) {
      benchmark<monolish::matrix::Dense<float>, float>(size, size, iter);
    }
    for (size_t size = DENSE_NN_BENCH_MIN; size <= DENSE_NN_BENCH_MAX;
         size += 1000) {
      benchmark<monolish::matrix::Dense<double>, double>(size, size, iter);
    }
  }

  // COO
  if ((strcmp(argv[1], "COO") == 0)) {
    for (size_t size = COO_NN_BENCH_MIN; size <= COO_NN_BENCH_MAX;
         size COO_NN_BENCH_ITER) {
      benchmark<monolish::matrix::COO<float>, float>(size, size, iter);
    }
    for (size_t size = COO_NN_BENCH_MIN; size <= COO_NN_BENCH_MAX;
         size COO_NN_BENCH_ITER) {
      benchmark<monolish::matrix::COO<double>, double>(size, size, iter);
    }
  }

  return 0;
}
