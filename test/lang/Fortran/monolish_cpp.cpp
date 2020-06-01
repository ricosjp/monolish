#include <iostream>
#include <omp.h>
#include"../../test_utils.hpp"
#include"../include/monolish_equation.hpp"
#include"../include/monolish_blas.hpp"


extern "C"{
	//int cfun_(int *vec, int* col, int* row, double *val)
	int monolish_spmv_(int *n, int* nnz, int row[], int col[], double val[], double xp[], double yp[])
	{
		int N = *n;
		int NNZ = *nnz;

		// 1-origin
		monolish::matrix::COO<double> COO(N, NNZ, row, col, val, 1);
		monolish::matrix::CRS<double> A(COO);

		monolish::vector x(xp, xp+N);
		monolish::vector y(xp, xp+N);

		monolish::util::send(A,x,y);

		monolish::blas::spmv(A, x, y);

		monolish::recv(y);

		return 0;
	}
}
