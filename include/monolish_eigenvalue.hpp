#pragma once
#include<omp.h>
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_common.hpp"

// NOT IMPL.
namespace monolish{
	namespace eigenvalue{

		class lanczos{
			int a;
		};
	}
}
