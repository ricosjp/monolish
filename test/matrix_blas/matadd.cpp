#include "../test_utils.hpp"
#include "monolish_blas.hpp"

#define FUNC "matadd"
#define DENSE_PERF 1 * M *N / time / 1.0e+9
#define CRS_PERF 2 * M *nnzrow / time / 1.0e+9

template <typename T>
void get_ans(const monolish::matrix::Dense<T> &A,
             const monolish::matrix::Dense<T> &B,
             monolish::matrix::Dense<T> &C) {

  if (A.get_row() != B.get_row()) {
    std::runtime_error("A.row != B.row");
  }
  if (A.get_col() != B.get_col()) {
    std::runtime_error("A.col != B.col");
  }

  // MN=MN+MN
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < A.get_nnz(); i++) {
    C.val[i] += A.val[i] * B.val[i];
  }
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test(const size_t M, const size_t N, double tol, int iter, int check_ans) {

  size_t nnzrow = 81;
  if ((nnzrow < M) && (nnzrow < N)) {
    nnzrow = 81;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedA);
  MAT_C C(seedA);

  if (check_ans == 1) {
    monolish::matrix::Dense<T> AA(seedA);
    monolish::matrix::Dense<T> BB(seedA);
    monolish::matrix::Dense<T> CC(seedA);

    get_ans(AA, BB, CC);
    monolish::matrix::COO<T> ansC(CC);

    monolish::util::send(A, B, C);
    monolish::blas::matadd(A, B, C);
    C.recv();

    monolish::matrix::COO<T> resultC(C);

    if (ans_check<T>(resultC.val.data(), ansC.val.data(), ansC.get_nnz(),
                     tol) == false) {
      return false;
    };

    A.device_free();
    B.device_free();
  }

  monolish::util::send(A, B, C);

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < iter; i++) {
    monolish::blas::matadd(A, B, C);
  }

  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  A.device_free();
  B.device_free();
  C.device_free();

  double time = sec / iter;
  std::cout << "func\tprec\tM\tN\ttime[sec]\tperf[GFLOPS] " << std::endl;
  std::cout << FUNC << "(" << C.type() << "=" << A.type() << "*" << C.type()
            << ")\t" << std::flush;
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

  if (argc != 9) {
    std::cout << "error $1: precision (double or float) \
            $2: format of A, $3: format of B, $4: format of C,\
            $5: M, $6: N, $7: iter, $8: error check (1/0)"
              << std::endl;
    return 1;
  }

  // MN=MK*KN
  const size_t M = atoi(argv[5]);
  const size_t N = atoi(argv[6]);
  int iter = atoi(argv[7]);
  int check_ans = atoi(argv[8]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (strcmp(argv[1], "double") == 0) {
    if ((strcmp(argv[2], "Dense") == 0) && (strcmp(argv[3], "Dense") == 0) &&
        (strcmp(argv[4], "Dense") == 0)) {
      if (test<monolish::matrix::Dense<double>, monolish::matrix::Dense<double>,
               monolish::matrix::Dense<double>, double>(M, N, 1.0e-6, iter,
                                                        check_ans) == false) {
        return 1;
      }
    }

    if ((strcmp(argv[2], "CRS") == 0) && (strcmp(argv[3], "CRS") == 0) &&
        (strcmp(argv[4], "CRS") == 0)) {
      if (test<monolish::matrix::CRS<double>, monolish::matrix::CRS<double>,
               monolish::matrix::CRS<double>, double>(M, N, 1.0e-6, iter,
                                                      check_ans) == false) {
        return 1;
      }
    }
  }

  if (strcmp(argv[1], "float") == 0) {
    if ((strcmp(argv[2], "Dense") == 0) && (strcmp(argv[3], "Dense") == 0) &&
        (strcmp(argv[4], "Dense") == 0)) {
      if (test<monolish::matrix::Dense<float>, monolish::matrix::Dense<float>,
               monolish::matrix::Dense<float>, float>(M, N, 1.0e-6, iter,
                                                      check_ans) == false) {
        return 1;
      }
    }

    if ((strcmp(argv[2], "CRS") == 0) && (strcmp(argv[3], "CRS") == 0) &&
        (strcmp(argv[4], "CRS") == 0)) {
      if (test<monolish::matrix::CRS<float>, monolish::matrix::CRS<float>,
               monolish::matrix::CRS<float>, float>(M, N, 1.0e-6, iter,
                                                    check_ans) == false) {
        return 1;
      }
    }
  }

  return 0;
}
