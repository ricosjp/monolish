#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_common.hpp"

namespace monolish{
	namespace equation{

		class solver{
			private:

			protected:
				int lib = 0;
				double tol = 1.0e-8;
				size_t miniter = SIZE_MAX;
				size_t maxiter = 0;
				size_t precon_num=0;
				
				/**
				 * @brief create q = Ap
				 **/
				int precon_init(matrix::CRS<double> &A, vector<double> &p, vector<double> &q);

				/**
				 * @brief apply q = Ap
				 **/
				int precon_apply(matrix::CRS<double> &A, vector<double> &p, vector<double> &q);

			public:

				/**
				 * @brief create solver class
				 * @param[in] 
				 **/
				 solver(){};

				/**
				 * @brief set library option
				 * @param[in] library number
				 **/
				void set_lib(double l){lib = l;}

				/**
				 * @brief set tolerance (default:1.0e-8)
				 * @param[in] tol tolerance
				 **/
				void set_tol(double t){tol = t;}

				/**
				 * @brief set max iter. (default = 0)
				 * @param[in] max maxiter
				 **/
				void set_maxiter(size_t max){maxiter = max;}

				/**
				 * @brief set min iter. (default = SIZE_MAX)
				 * @param[in] min miniter
				 **/
				void set_miniter(size_t min){miniter = min;}

				/**
				 * @brief set precon number
				 * @param[in] precondition number (0:none, 1:jacobi)
				 **/
				void set_precon(size_t precondition){precon_num = precondition;}


				///////////////////////////////////////////////////////////////////

				/**
				 * @brief get library option
				 * @return library number
				 **/
				int get_lib(){return lib;}

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
				 * @brief get miniter
				 * @return  miniter
				 **/
				size_t get_miniter(){return miniter;}

				/**
				 * @brief get precondition number
				 * @return  precondition number
				 **/
				size_t get_precon(){return precon_num;}

		};

		/**
		 * @brief CG solver class
		 */
		class cg : public solver{
			private:
				int monolish_cg(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

			public:
				using solver::solver;

				/**
				 * @brief solve Ax = b by cg method(lib=0: monolish)
				 * @param[in] A CRS format Matrix
				 * @param[in] x solution vector
				 * @param[in] b right hand vector
				 * @return error code (only 0 now)
				 **/
				int solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
		};


		// only external
		class LU : public solver{
			private:
				using solver::solver;
				int lib = 1; // lib is 1
				int mumps_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int cusolver_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

			public:
				/**
				 * @brief solve Ax = b by LU method(lib=0: MUMPS(NOT IMPL), lib=1: cusolver)
				 * @param[in] A CRS format Matrix
				 * @param[in] x solution vector
				 * @param[in] b right hand vector
				 * @return error code (only 0 now)
				 **/
				int solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
		};

		//jacobi////////////////////////////////
		class jacobi : public solver{
			private:
				int monolish_jacobi(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int monolish_Pjacobi(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
			public:
				using solver::solver;

				/**
				 * @brief solve Ax = b by jacobi method(lib=0: monolish)
				 * @param[in] A CRS format Matrix
				 * @param[in] x solution vector
				 * @param[in] b right hand vector
				 * @return error code (only 0 now)
				 **/
				int solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

				int Pinit(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int Papply(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
		};
	}
}
