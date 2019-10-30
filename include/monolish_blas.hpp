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
		double dot(monolish::vector<double> &x, monolish::vector<double> &y);
		double axpy(monolish::vector<double> &x, monolish::vector<double> &y);

		float axpy(monolish::vector<float> &x, monolish::vector<float> &y);
	}
}
