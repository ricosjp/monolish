#pragma once
#include<omp.h>
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

namespace monolish{

	template<typename MatrixFloat, typename Intger>
		class COO_matrix{
			std::vector<MatrixFloat> val;
			std::vector<MatrixFloat> col;
			std::vector<MatrixFloat> row;

			void input(const char* filename);
			void at(int i, int j);
			void inseart(const char* filename);
		};

	
}
