#include "../../test_utils.hpp"
#include "../include/monolish_blas.hpp"
#include "../include/monolish_equation.hpp"
#include <iostream>
#include <omp.h>

extern "C" {
// int cfun_(int *vec, int* col, int* row, double *val)
int monolish_matvec_(int *n, int *nnz, int row[], int col[], double val[],
                     double xp[], double yp[]) {
  int N = *n;
  int NNZ = *nnz;

  // 1-origin
  monolish::matrix::COO<double> COO(N, N, NNZ, row, col, val, 1);
  monolish::matrix::CRS<double> A(COO);

  monolish::vector<double> x(xp, xp + N);
  monolish::vector<double> y(xp, xp + N);

  monolish::util::send(A, x, y);

  monolish::blas::matvec(A, x, y);

  monolish::util::recv(y);

  return 0;
}
}
