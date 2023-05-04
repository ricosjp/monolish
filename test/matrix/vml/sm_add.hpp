#include "../../test_utils.hpp"

template <typename T>
void ans_sm_add(const monolish::matrix::Dense<T> &A, const T alpha,
                monolish::matrix::Dense<T> &C) {

  for (int i = 0; i < A.get_nnz(); i++) {
    C.begin()[i] = A.begin()[i] + alpha;
  }
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_sm_add(const size_t M, const size_t N, double tol) {

  monolish::matrix::Dense<T> seed(M, N, 1.0, 2.0);
  monolish::matrix::COO<T> seedA(seed);

  MAT_A A(seedA);
  T alpha = 123.0;
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_sm_add(AA, alpha, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::vml::add(A, alpha, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_sm_add(const size_t M, const size_t N, double tol) {

  monolish::matrix::Dense<T> seed(M, N, 1.0, 2.0);
  monolish::matrix::COO<T> seedA(seed);

  MAT_A A(seedA);
  T alpha = 123.0;
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_sm_add(AA, alpha, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::vml::add(A, alpha, C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}
