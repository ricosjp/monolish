#include "../../test_utils.hpp"

template <typename T>
void ans_tscal(const double alpha, monolish::tensor::tensor_Dense<T> &A) {

  for (int i = 0; i < A.get_nnz(); i++)
    A.begin()[i] = alpha * A.begin()[i];
}

template <typename MAT, typename T>
bool test_send_tscal(const size_t M, const size_t N, const size_t L,
                     double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }
  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  T alpha = 123.0;
  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_tscal(alpha, AA);
  monolish::tensor::tensor_COO<T> ansA(AA);

  A.send();
  monolish::blas::tscal(alpha, A);
  A.recv();
  monolish::tensor::tensor_COO<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.begin(), ansA.begin(),
                      ansA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_tscal(const size_t M, const size_t N, const size_t L, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }
  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  T alpha = 123.0;
  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_tscal(alpha, AA);
  monolish::tensor::tensor_COO<T> ansA(AA);

  monolish::blas::tscal(alpha, A);
  monolish::tensor::tensor_COO<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.begin(), ansA.begin(),
                      ansA.get_nnz(), tol);
}
