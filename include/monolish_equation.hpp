#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_common.hpp"

namespace monolish{
	namespace equation{
	/**
	* @brief CG solver class
	*/
		class cg{
			private:
				int lib = 0;
				double tol = 1.0e-8;
				int maxiter;

				void monolish_cg(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

			public:
				cg(){}

				/**
				 * @brief solve Ax = b by cg method
				 * @param[in] A vector length
				 * @param[in] x solution vector
				 * @param[in] b right hand vector
				 * @return error code (0 or 1 now)
				 **/
				void solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				//int solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

				/**
				 * @brief set tolerance (default)
				 * @param[in] tol tolerance
				 **/
				void set_tol(double t){tol = t;}

				/**
				 * @brief set max iteration
				 * @param[in] max iteration
				 **/
				void set_maxiter(int max){maxiter = max;}

				/**
				 * @brief get tolerance
				 * @return tolerance
				 **/
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
