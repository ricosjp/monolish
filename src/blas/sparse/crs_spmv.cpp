#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#include "../../../include/monolish_blas.hpp"

#define BENCHMARK
namespace monolish{

	void blas::spmv(matrix::CRS<double> &A, vector<double> &x, vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size() || A.get_row() != (int)x.size()){
			throw std::runtime_error("error vector size is not same");

		}

#if USE_GPU // gpu
		int n = A.get_row();
		int nnz = A.get_nnz();
		double* xd = x.data();
		double* yd = y.data();

		double* vald = A.val.data();
		int* rowd = A.row_ptr.data();
		int* cold = A.col_ind.data();

		#pragma acc data pcopyin(xd[0:n], vald[0:nnz], rowd[0:n+1], cold[0:nnz]) copyout(yd[0:n]) 
		{
			#pragma acc kernels
			{
				#pragma acc loop independent 
				for(int i = 0 ; i < n; i++){
					yd[i] = 0;
				}

				#pragma acc loop independent
				for(int i = 0 ; i < n; i++){
					for(int j = rowd[i] ; j < rowd[i+1]; j++){
						yd[i] += vald[j] * xd[cold[j]];
					}
				}
			}
		}

#else // cpu

	#pragma omp parallel for 
		for(int i = 0 ; i < A.get_row(); i++)
			y.val[i] = 0;

	#pragma omp parallel for
		for(int i = 0 ; i < A.get_row(); i++)
			for(int j = A.row_ptr[i] ; j < A.row_ptr[i+1]; j++)
				y.val[i] += A.val[j] * x.val[A.col_ind[j]];

#endif

		logger.func_out();
	}
}


