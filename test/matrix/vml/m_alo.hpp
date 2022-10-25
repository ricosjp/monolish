#include "../../test_utils.hpp"

template <typename T>
void ans_m_alo(const monolish::matrix::Dense<T> &A, const T alpha, const T beta,
               monolish::matrix::Dense<T> &C) {

  // MN=MN+MN
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < A.get_nnz(); i++) {
    if (A.val[i] > 0) {
      C.val[i] = alpha * A.val[i];
    } else {
      C.val[i] = beta * A.val[i];
    }
  }
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_m_alo(const size_t M, const size_t N, double tol) {

  monolish::matrix::Dense<T> seed(M, N, -1.0, 1.0);
  monolish::matrix::COO<T> seedA(seed);

  MAT_A A(seedA);
  T alpha = 1.5;
  T beta = 0.5;
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_m_alo(AA, alpha, beta, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::vml::alo(A, alpha, beta, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_m_alo(const size_t M, const size_t N, double tol) {

  monolish::matrix::Dense<T> seed(M, N, -1.0, 1.0);
  monolish::matrix::COO<T> seedA(seed);

  MAT_A A(seedA);
  T alpha = 1.5;
  T beta = 0.5;
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_m_alo(AA, alpha, beta, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::vml::alo(A, alpha, beta, C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}
