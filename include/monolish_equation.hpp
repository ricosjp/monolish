#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_matrix.hpp"

namespace monolish{
	namespace equation{

		class cg{
			private:
			public:
			cg(){}
			int a;
			void test_func();
			int test_func2(){
				return 2;
			}
			void solve();
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
