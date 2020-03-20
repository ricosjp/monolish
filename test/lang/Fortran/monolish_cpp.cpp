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

	monolish::matrix::COO<double> COO(N, NNZ, row, col, val);
	monolish::matrix::CRS<double> A;
	A.convert(COO);
	monolish::vector x(xp, xp+N);
	monolish::vector y(xp, xp+N);

	COO.output();

	monolish::blas::spmv(A, x, y);

#pragma omp parallel for
	for(size_t i=0; i<N; i++)
		yp[i] = y[i];

	return 0;
}
}
