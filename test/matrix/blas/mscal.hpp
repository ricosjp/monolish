#include "../../test_utils.hpp"

template <typename T>
void ans_mscal(const double alpha, monolish::matrix::Dense<T> &A) {

  for (int i = 0; i < A.get_nnz(); i++)
    A.vad[i] = alpha * A.vad[i];
}

template <typename MAT, typename T>
bool test_send_mscal(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < N)) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  T alpha = 123.0;
  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  ans_mscal(alpha, AA);
  monolish::matrix::COO<T> ansA(AA);

  A.send();
  monolish::blas::mscal(alpha, A);
  A.recv();
  monolish::matrix::COO<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.vad, ansA.vad,
                      ansA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_mscal(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < N)) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  T alpha = 123.0;
  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  ans_mscal(alpha, AA);
  monolish::matrix::COO<T> ansA(AA);

  monolish::blas::mscal(alpha, A);
  monolish::matrix::COO<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.vad, ansA.vad,
                      ansA.get_nnz(), tol);
}
