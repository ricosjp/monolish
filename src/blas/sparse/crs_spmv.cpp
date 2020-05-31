#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#include "../../../include/monolish_blas.hpp"

namespace monolish{

	void blas::spmv(matrix::CRS<double> &A, vector<double> &x, vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size() || A.get_row() != (size_t)x.size()){
			throw std::runtime_error("error vector size is not same");

		}

		size_t n = A.get_row();
		size_t nnz = A.get_nnz();
		double* xd = x.data();
		double* yd = y.data();

		double* vald = A.val.data();
		int* rowd = A.row_ptr.data();
		int* cold = A.col_ind.data();

		#if USE_GPU // gpu

		#pragma acc data present(xd[0:n], yd[0:n], vald[0:nnz], rowd[0:n+1], cold[0:nnz])
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
		#else // cpu

		#pragma omp parallel for 
			for(size_t i = 0 ; i < A.get_row(); i++)
				yd[i] = 0;
		#pragma omp parallel for
		for(int i = 0 ; i < (int)A.get_row(); i++)
			for(int j = A.row_ptr[i] ; j < A.row_ptr[i+1]; j++)
				yd[i] += vald[j] * xd[A.col_ind[j]];

		#endif

		logger.func_out();
	}
	void blas::spmv(matrix::CRS<float> &A, vector<float> &x, vector<float> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size() || A.get_row() != (size_t)x.size()){
			throw std::runtime_error("error vector size is not same");

		}

		size_t n = A.get_row();
		size_t nnz = A.get_nnz();
		float* xd = x.data();
		float* yd = y.data();

		float* vald = A.val.data();
		int* rowd = A.row_ptr.data();
		int* cold = A.col_ind.data();

		#if USE_GPU // gpu

		#pragma acc data present(xd[0:n], yd[0:n], vald[0:nnz], rowd[0:n+1], cold[0:nnz])
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
		#else // cpu

		#pragma omp parallel for 
			for(size_t i = 0 ; i < A.get_row(); i++)
				yd[i] = 0;
		#pragma omp parallel for
		for(int i = 0 ; i < (int)A.get_row(); i++)
			for(int j = A.row_ptr[i] ; j < A.row_ptr[i+1]; j++)
				yd[i] += vald[j] * xd[A.col_ind[j]];

		#endif

		logger.func_out();
	}
}
