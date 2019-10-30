#pragma once
#include<omp.h>
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

namespace monolish{

	template<typename Float, typename Intger>
		class COO_matrix{
			std::vector<Float> val;
			std::vector<Float> col;
			std::vector<Float> row;

			void input(const char* filename);
			void at(int i, int j);
			void inseart(const char* filename);
		};

	template<typename Float, typename Intger>
		class vector{
			std::vector<Float> val;
		};

	
}
