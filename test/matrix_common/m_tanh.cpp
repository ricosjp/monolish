#include "../test_utils.hpp"
#include "monolish_blas.hpp"

#define FUNC "m_tanh"
#define DENSE_PERF M *N / time / 1.0e+9
#define CRS_PERF M *nnzrow / time / 1.0e+9

template <typename T> void get_ans(monolish::matrix::Dense<T> &A) {

  for (size_t i = 0; i < A.get_nnz(); i++) {
    A.val[i] = std::tanh(A.val[i]);
  }
}

template <typename MAT, typename T>
bool test(const size_t M, const size_t N, double tol, int iter, int check_ans) {
  size_t nnzrow = 81;
  if (nnzrow < N) {
    nnzrow = 81;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  if (check_ans == 1) {
    monolish::matrix::Dense<T> AA(seedA);
    get_ans(AA);

    A.send();
    A.tanh();
    A.recv();
    monolish::matrix::Dense<T> resultA(A);

    if (ans_check<T>(resultA.val.data(), AA.val.data(), AA.get_nnz(), tol) ==
        false) {
      return false;
    };
    A.device_free();
  }

  A.send();

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < iter; i++) {
    A.tanh();
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  A.device_free();

  double time = sec / iter;
  std::cout << "func\tprec\tM\tN\ttime[sec]\tperf[GFLOPS] " << std::endl;
  std::cout << FUNC << "(" << A.type() << ")\t" << std::flush;
  std::cout << get_type<T>() << "\t" << std::flush;
  std::cout << M << "\t" << std::flush;
  std::cout << N << "\t" << std::flush;
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

  if (argc != 7) {
    std::cout << "error $1: precision (double or float) $2: format, $3: row, "
                 "$4: col, $5: iter, $6: error check (1/0)"
              << std::endl;
    return 1;
  }

  const size_t M = atoi(argv[3]);
  const size_t N = atoi(argv[4]);
  int iter = atoi(argv[5]);
  int check_ans = atoi(argv[6]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  // exec and error check
  if (strcmp(argv[1], "double") == 0) {
    if ((strcmp(argv[2], "Dense") == 0)) {
      if (test<monolish::matrix::Dense<double>, double>(M, N, 1.0e-6, iter,
                                                        check_ans) == false) {
        return 1;
      }
    }
    if ((strcmp(argv[2], "CRS") == 0)) {
      if (test<monolish::matrix::CRS<double>, double>(M, N, 1.0e-6, iter,
                                                      check_ans) == false) {
        return 1;
      }
    }
  }

  if (strcmp(argv[1], "float") == 0) {
    if ((strcmp(argv[2], "Dense") == 0)) {
      if (test<monolish::matrix::Dense<float>, float>(M, N, 1.0e-6, iter,
                                                      check_ans) == false) {
        return 1;
      }
    }
    if ((strcmp(argv[2], "CRS") == 0)) {
      if (test<monolish::matrix::CRS<float>, float>(M, N, 1.0e-6, iter,
                                                    check_ans) == false) {
        return 1;
      }
    }
  }

  return 0;
}
