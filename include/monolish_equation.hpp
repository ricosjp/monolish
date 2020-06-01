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
				size_t miniter = 0;
				size_t maxiter = SIZE_MAX;
				size_t precon_num=0;
				size_t resid_method=0;
				bool print_rhistory = false;
				std::string rhistory_file;
				std::ostream* rhistory_stream;
				
				/**
				 * @brief create q = Ap
				 **/
				int precon_init(matrix::CRS<double> &A, vector<double> &p, vector<double> &q);

				/**
				 * @brief apply q = Ap
				 **/
				int precon_apply(matrix::CRS<double> &A, vector<double> &p, vector<double> &q);

				double get_residual(vector<double>& x);
				float get_residual(vector<float>& x);

			public:

				/**
				 * @brief create solver class
				 * @param[in] 
				 **/
				 solver(){};

				/**
				 * @brief delete solver class
				 * @param[in] 
				 **/
				 ~solver(){
					 if(rhistory_stream != &std::cout && rhistory_file.empty() != true){
						 delete rhistory_stream;
					 }
				 }

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
				 * @brief set max iter. (default = SIZE_MAX)
				 * @param[in] max maxiter
				 **/
				void set_maxiter(size_t max){maxiter = max;}

				/**
				 * @brief set min iter. (default = 0)
				 * @param[in] min miniter
				 **/
				void set_miniter(size_t min){miniter = min;}

				/**
				 * @brief set precon number
				 * @param[in] precondition number (0:none, 1:jacobi)
				 **/
				void set_precon(size_t precondition){precon_num = precondition;}

				/**
				 * @brief set residual method(default=0)
				 * @param[in] precondition number (0:nrm2)
				 **/
				void set_residual_method(size_t precondition){precon_num = precondition;}

				/**
				 * @brief print rhistory to standart out true/false. (default = false)
				 * @param[in] flag 
				 **/
				void set_print_rhistory(bool flag){
					print_rhistory=flag;
					rhistory_stream = &std::cout;
				}

				/**
				 * @brief rhistory filename
				 * @param[in] file: output file name
				 **/
				void set_rhistory_filename(std::string file){
					rhistory_file = file;

					//file open
					rhistory_stream	= new std::ofstream(rhistory_file);
					if(rhistory_stream -> fail()){
						throw std::runtime_error("error bad filename");
					}
				}
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

				/**
				 * @brief get residual method(default=0)
				 * @return residual method number
				 **/
				size_t get_residual_method(){return resid_method;}

				/**
				 * @brief get print rhistory status
				 * @param[in] print rhistory true/false
				 **/
				bool get_print_rhistory(){return print_rhistory;}

		};

		/**
		 * @brief CG solver class
		 */
		class CG : public solver{
			private:
				template<typename T>
				int monolish_CG(matrix::CRS<T> &A, vector<T> &x, vector<T> &b);

			public:
				using solver::solver;

				/**
				 * @brief solve Ax = b by CG method(lib=0: monolish)
				 * @param[in] A CRS format Matrix
				 * @param[in] x solution vector
				 * @param[in] b right hand vector
				 * @return error code (only 0 now)
				 **/
				template<typename T> int solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b);
		};


		/**
		 * @brief LU solver class (does not impl. now)
		 */
		class LU : public solver{
			private:
				using solver::solver;
				int lib = 1; // lib is 1
				int mumps_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int cusolver_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int singularity;
				int reorder;

			public:
				void set_reorder(int r){ reorder = r; }

				int get_sigularity(){ return singularity; }

				int solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
		};

		/**
		 * @brief QR solver class (GPU only now). can use set_tol(), get_til(),
		 * set_reorder(), get_singularity(). default reorder algorithm is csrmetisnd
		 */
		class QR : public solver{
			private:
				using solver::solver;
				int lib = 1; // lib is 1
				int cusolver_QR(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int cusolver_QR(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
				int singularity;
				int reorder=3;

			public:

				/**
				 * @brief 0: no ordering 1: symrcm, 2: symamd, 3: csrmetisnd is used to reduce zero fill-in.
				*/
				void set_reorder(int r){ reorder = r; }

				/**
				 * @brief -1 if A is symmetric postive definite.
				*/
				int get_sigularity(){ return singularity; }
				
				template<typename T> int solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b);
		};

		/**
		 * @brief Cholesky solver class (GPU only now). can use set_tol(), get_til(),
		 * set_reorder(), get_singularity(). default reorder algorithm is csrmetisnd
		 */
		class Cholesky : public solver{
			private:
				using solver::solver;
				int lib = 1; // lib is 1
				int cusolver_Cholesky(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int cusolver_Cholesky(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
				int singularity;
				int reorder=3;

			public:

				/**
				 * @brief 0: no ordering 1: symrcm, 2: symamd, 3: csrmetisnd is used to reduce zero fill-in.
				*/
				void set_reorder(int r){ reorder = r; }

				/**
				 * @brief -1 if A is symmetric postive definite.
				*/
				int get_sigularity(){ return singularity; }
				
				template<typename T> int solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b);
		};

		//jacobi////////////////////////////////
		class Jacobi : public solver{
			private:
				int monolish_Jacobi(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int monolish_PJacobi(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
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
