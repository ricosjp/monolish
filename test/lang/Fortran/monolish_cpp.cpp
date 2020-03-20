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
	monolish::matrix::CRS<double> A(COO);

	monolish::vector x(xp, xp+N);
	monolish::vector y(xp, xp+N);

	monolish::blas::spmv(A, x, y);

#pragma omp parallel for
	for(size_t i=0; i<N; i++)
		yp[i] = y[i];


	// omake: sparse LU (cant solve on cpu now)////////
	
	monolish::equation::LU LU_solver;

	monolish::vector<double> slu_x(A.get_row(), 0.0);
	monolish::vector<double> slu_b(A.get_row(), 0.0, 1.0); //rand(0~1)

	LU_solver.solve(A, slu_x, slu_b);

	//slu_x.print_all();
	

	// omake: end////////
	return 0;
}
}
