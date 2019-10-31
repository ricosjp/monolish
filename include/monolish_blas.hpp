#pragma once
#include<vector>
#include"common/monolish_common.hpp"
#include<cblas.h>
#include<omp.h>
#include<stdio.h>

#if defined USE_MPI
#include<mpi.h>
#endif

namespace monolish{
	namespace blas{

// toriaezu double dake
		double dot(vector<double> &x, vector<double> &y);

		void spmv(CRS_matrix<double> &A, vector<double> &x, vector<double> &y);


	}
}
