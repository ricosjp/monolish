#include "../../test_utils.hpp"

template <typename T>
void ans_sm_max(const monolish::matrix::Dense<T> &A, const T alpha,
                monolish::matrix::Dense<T> &C) {

  // MN=MN+MN
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < A.get_nnz(); i++) {
    C.begin()[i] = std::max(A.begin()[i], alpha);
  }
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_sm_max(const size_t M, const size_t N, double tol) {

  monolish::matrix::Dense<T> seed(M, N, 1.0, 2.0);
  monolish::matrix::COO<T> seedA(seed);

  MAT_A A(seedA);
  T alpha = 1.5;
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_sm_max(AA, alpha, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::vml::max(A, alpha, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_sm_max(const size_t M, const size_t N, double tol) {

  monolish::matrix::Dense<T> seed(M, N, 1.0, 2.0);
  monolish::matrix::COO<T> seedA(seed);

  MAT_A A(seedA);
  T alpha = 1.5;
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_sm_max(AA, alpha, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::vml::max(A, alpha, C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}
