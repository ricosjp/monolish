#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_common.hpp"

namespace monolish{
	namespace equation{
		class cg{
			private:
				int optionA = 1;
				double tol = 1.0e-8;
				int maxiter=100;


			public:
				cg(){}
				int a;
				void monolish_cg(vector<double> &x, vector<double> b);
				void solve(vector<double> &x, vector<double> b);
		};

		class jacobi{
			private:
			public:
				int a;
				void test_func();
		};

		class ilu{
			private:
			public:
				int a;
				void test_func();
		};

	}
}
