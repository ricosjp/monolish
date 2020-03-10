#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_common.hpp"

namespace monolish{
	namespace equation{
//cg///////////////////////
		class cg{
			private:
				int lib = 0;
				double tol = 1.0e-8;
				int maxiter;

				void monolish_cg(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

			public:
				cg(){}

				void solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

				void set_tol(double t){tol = t;}
				void set_maxiter(int it){maxiter = it;}

				double get_tol(){return tol;}
				int get_maxiter(){return maxiter;}
		};

//jacobi////////////////////////////////
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

		// only external
		class LU{
			private:
				int lib = 1;
				void mumps_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				void cusolver_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

			public:
				void solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
		};
	}
}
