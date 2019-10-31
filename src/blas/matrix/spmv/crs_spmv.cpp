#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#include<cblas.h>
#include "../../../../include/monolish_blas.hpp"

#define BENCHMARK
namespace monolish{

	void blas::spmv(CRS_matrix<double> &A, vector<double> &x, vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size() || A.get_row() != (int)x.size()){
			throw std::runtime_error("error vector size is not same");

		}

#if USE_GPU

#else

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


