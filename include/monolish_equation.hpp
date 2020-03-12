#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_common.hpp"

namespace monolish{
	namespace equation{

		//jacobi////////////////////////////////
		class jacobi{
			private:
			public:
				int a;
				void test_func();
				jacobi(){};

				void solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
		};

		/**
		 * @brief CG solver class
		 */
		class cg{
			private:
				int lib = 0;
				double tol = 1.0e-8;
				size_t maxiter;
				size_t precon_num=0;

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

				/**
				 * @brief set tolerance (default:1.0e-8)
				 * @param[in] tol tolerance
				 **/
				void set_tol(double t){tol = t;}

				/**
				 * @brief set max iteration
				 * @param[in] max iteration
				 **/
				void set_maxiter(size_t max){maxiter = max;}

				/**
				 * @brief set precon
				 * @param[in] precondition number (0:none, 1:jacobi)
				 **/
				void set_precon(size_t precondition){precon_num = precondition;}

				/**
				 * @brief get tolerance
				 * @return tolerance
				 **/
				double get_tol(){return tol;}

				/**
				 * @brief get maxiter
				 * @return  maxiter
				 **/
				size_t get_maxiter(){return maxiter;}

				/**
				 * @brief get precondition number
				 * @return  maxiter
				 **/
				size_t get_maxprecon(){return precon_num;}
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
