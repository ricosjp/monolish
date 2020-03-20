#include <iostream>
#include"../../test_utils.hpp"
#include"../include/monolish_equation.hpp"
#include"../include/monolish_blas.hpp"


extern "C"{
//int cfun_(int *vec, int* col, int* row, double *val)
int monolish_spmv_(int *n, int* nnz, int row[], int col[], double val[], double x[], double y[])
{
	int N = *n;
	y[0] = 1000;
	y[1] = 12345;
// 	double x = xp[1];
//
// 	printf("This is in C function...\n");
// 	for(int i=0; i<*n; i++)
// 	printf("i = %d, x = %f\n", N, xp[i]);
	return 0;
}
}
