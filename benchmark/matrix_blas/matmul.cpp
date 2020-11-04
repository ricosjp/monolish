#include "../benchmark_utils.hpp"

#define FUNC "matmul"
#define DENSE_PERF 2 * M *N *K / time / 1.0e+9
#define CRS_PERF 2 * M *N *nnzrow / time / 1.0e+9

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool benchmark(const size_t M, const size_t N, const size_t K, size_t iter) {

  size_t nnzrow = 81;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K)) {
    nnzrow = 81;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::band_matrix<T>(M, K, nnzrow, 1.0, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::band_matrix<T>(K, N, nnzrow, 1.0, 1.0);
  monolish::matrix::COO<T> seedC =
      monolish::util::band_matrix<T>(M, N, nnzrow, 1.0, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::util::send(A, B, C);

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < iter; i++) {
    monolish::blas::matmul(A, B, C);
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  A.device_free();
  B.device_free();
  C.device_free();

  double time = sec / iter;
  std::cout << FUNC << "(" << C.type() << "," << A.type() << "," << C.type()
            << ")\t" << std::flush;
  std::cout << get_type<T>() << "\t" << std::flush;
  std::cout << M << "\t" << std::flush;
  std::cout << N << "\t" << std::flush;
  std::cout << K << "\t" << std::flush;
  std::cout << time << "\t" << std::flush;

  if ((strcmp(A.type().data(), "Dense") == 0)) {
    std::cout << DENSE_PERF << "\t" << std::endl;
  }

  if ((strcmp(A.type().data(), "CRS") == 0)) {
    std::cout << CRS_PERF << "\t" << std::endl;
  }

  return true;
}

int main(int argc, char **argv) {

  if (argc <= 3) {
    std::cout << "error $1: format of A, $2: format of B, $3: format of C" << std::endl;
    return 1;
  }

  std::cout << "func\tprec\tM\tN\tK\ttime[sec]\tperf[GFLOPS] " << std::endl;

  size_t iter = MATRIX_BENCH_ITER;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");
  
  if (argc == 8) {

    // MN=MK*KN
    const size_t M = atoi(argv[5]);
    const size_t N = atoi(argv[6]);
    const size_t K = atoi(argv[7]);

    if (strcmp(argv[4], "double") == 0) {
      if ((strcmp(argv[1], "Dense") == 0) && (strcmp(argv[2], "Dense") == 0) &&
          (strcmp(argv[3], "Dense") == 0)) {
        benchmark<monolish::matrix::Dense<double>, monolish::matrix::Dense<double>,
          monolish::matrix::Dense<double>, double>(M, N, K, iter);
      }

      if ((strcmp(argv[1], "CRS") == 0) && (strcmp(argv[2], "Dense") == 0) &&
          (strcmp(argv[3], "Dense") == 0)) {
        benchmark<monolish::matrix::CRS<double>, monolish::matrix::Dense<double>,
          monolish::matrix::Dense<double>, double>(M, N, K, iter);
      }
    }

    if (strcmp(argv[4], "float") == 0) {
      if ((strcmp(argv[1], "Dense") == 0) && (strcmp(argv[2], "Dense") == 0) &&
          (strcmp(argv[3], "Dense") == 0)) {
        benchmark<monolish::matrix::Dense<float>, monolish::matrix::Dense<float>,
          monolish::matrix::Dense<float>, float>(M, N, K, iter);
      }

      if ((strcmp(argv[1], "CRS") == 0) && (strcmp(argv[2], "Dense") == 0) &&
          (strcmp(argv[3], "Dense") == 0)) {
        benchmark<monolish::matrix::CRS<float>, monolish::matrix::Dense<float>,
          monolish::matrix::Dense<float>, float>(M, N, K, iter);
      }
    }
  }

  //Dense,Dense,Dense
  if ((strcmp(argv[1], "Dense") == 0) && (strcmp(argv[2], "Dense") == 0) &&
      (strcmp(argv[3], "Dense") == 0)) {
    for(size_t size = DENSE_NNN_BENCH_MIN; size <= DENSE_NNN_BENCH_MAX; size += 1000){
      benchmark<monolish::matrix::Dense<float>, monolish::matrix::Dense<float>,
        monolish::matrix::Dense<float>, float>(size, size, size, iter);
    }
    for(size_t size = DENSE_NNN_BENCH_MIN; size <= DENSE_NNN_BENCH_MAX; size += 1000){
      benchmark<monolish::matrix::Dense<double>, monolish::matrix::Dense<double>,
        monolish::matrix::Dense<double>, double>(size, size, size, iter);
    }
  }

  //Dense,CRS,Dense
  if ((strcmp(argv[1], "CRS") == 0) && (strcmp(argv[2], "Dense") == 0) &&
      (strcmp(argv[3], "Dense") == 0)) {
    for(size_t size = CRS_NNN_BENCH_MIN; size <= CRS_NNN_BENCH_MAX; size += 1000){
      benchmark<monolish::matrix::CRS<float>, monolish::matrix::Dense<float>,
        monolish::matrix::Dense<float>, float>(size, size, size, iter);
    }
    for(size_t size = CRS_NNN_BENCH_MIN; size <= CRS_NNN_BENCH_MAX; size += 1000){
      benchmark<monolish::matrix::CRS<double>, monolish::matrix::Dense<double>,
        monolish::matrix::Dense<double>, double>(size, size, size, iter);
    }
  }

  return 0;
}
