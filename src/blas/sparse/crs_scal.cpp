#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#include "../../../include/monolish_blas.hpp"

namespace monolish{

	void blas::mscal(const double alpha, matrix::CRS<double> &A){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		size_t n = A.get_row();
		size_t nnz = A.get_nnz();

		double* vald = A.val.data();
		int* rowd = A.row_ptr.data();
		int* cold = A.col_ind.data();

#if USE_GPU // gpu
		#pragma acc data pcopy(vald[0:nnz]) 
		{
			#pragma acc kernels
			{
				#pragma acc loop independent 
				for(int i = 0 ; i < nnz; i++){
					vald[i] = alpha * vald[i];
				}
			}
		}

#else // cpu

	#pragma omp parallel for 
		for(size_t i = 0 ; i < nnz; i++)
			vald[i] = alpha * vald[i];
#endif

		logger.func_out();
	}
}
