#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

//#include"common/monolish_matrix.hpp"

namespace monolish{

	namespace blas{

		extern double dot(std::vector<double> x, std::vector<double> y);
		extern double dot(std::vector<float> x, std::vector<float> y);

	}
}
