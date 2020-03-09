#pragma once
#include<vector>
#include"common/monolish_common.hpp"
#include<omp.h>
#include<stdio.h>

#if defined USE_MPI
#include<mpi.h>
#endif

namespace monolish{
	namespace blas{

////////////vector////////////////////

		//axpy
  		void axpy(const double alpha, const vector<double> &x, vector<double> &y);

		//axpyz
  		void axpyz(const double alpha, const vector<double> &x, const vector<double> &y, vector<double> &z);

		//dot
  		double dot(const vector<double> &x, const vector<double> &y);
  		float dot(const vector<float> &x,const vector<float> &y);

  		double nrm2(vector<double> &x, vector<double> &y);


  		void xpay(vector<double> &x, vector<double> &y);

  		void scal(vector<double> &x, vector<double> &y);

  		void copy(vector<double> &x, vector<double> &y);

////////////Sparse Matrix/////////////
		void spmv(matrix::CRS<double> &A, vector<double> &x, vector<double> &y);

	}
}
