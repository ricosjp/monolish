#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_common.hpp"
#include<functional>

namespace monolish{
	namespace equation{

		class none{
			public:
			template<typename T> void precon_create(matrix::CRS<T>& A);
			template<typename T> void precon_apply(const vector<T>& r, vector<T>& z);
			template<typename T> int solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b);
		};

		template<typename Float>
		class precon{
			private:
			public:
				vector<Float> M;
				monolish::matrix::CRS<Float> A;
				std::function<void(matrix::CRS<Float>&)> precon_create;
				std::function<void(const vector<Float>& r, vector<Float>& z)> precon_apply;

				std::function<void(void)> get_precon();

				void set_precon_data(vector<Float>& m){M=m;};
				vector<Float> get_precon_data() {return M;};

				precon(){
					equation::none p;
					precon_create = std::bind(&none::precon_create<Float>, &p, std::placeholders::_1);
					precon_apply = std::bind(&none::precon_apply<Float>, &p, std::placeholders::_1, std::placeholders::_2);
				}
		};

		class solver{
			private:

			protected:

				int lib = 0;
				double tol = 1.0e-8;
				size_t miniter = 0;
				size_t maxiter = SIZE_MAX;
				size_t resid_method=0;
				bool print_rhistory = false;
				std::string rhistory_file;
				std::ostream* rhistory_stream;
				
				double get_residual(vector<double>& x);
				float get_residual(vector<float>& x);

				// for mixed prec.
				precon<float> f_precon;
				precon<double> d_precon;

				void precon_create(matrix::CRS<double> &A);
				void precon_create(matrix::CRS<float> &A);
				void precon_apply(const vector<float>& r, vector<float>& z);
				void precon_apply(const vector<double>& r, vector<double>& z);

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
				 * @brief set precondition create fucntion
				 * @param[in] f function 
				 **/
				 void set_precon_create(std::function<void(matrix::CRS<double>&)> f);
				 void set_precon_create(std::function<void(matrix::CRS<float>&)> f);

				/**
				 * @brief set precondition apply fucntion
				 * @param[in] f function 
				 **/
				 void set_precon_apply(std::function<void(const vector<double>& z, vector<double>& r)> f);
				 void set_precon_apply(std::function<void(const vector<float>& z, vector<float>& r)> f);

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
				 * @brief set residual method (default=0)
				 * @param[in] r residualt method number (0:nrm2)
				 **/
				void set_residual_method(size_t r){resid_method = r;}

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

// 				template<typename T> void precon_create(matrix::CRS<T> A)
// 				{throw std::runtime_error("error, CG cant be used as precon");}
// 				template<typename T> void precon_apply(vector<T> z, vector<T> r)
// 				{throw std::runtime_error("error, CG cant be used as precon");}
		};

		//jacobi////////////////////////////////
		class Jacobi : public solver{
			private:
				int monolish_Jacobi(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
				int monolish_PJacobi(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
			public:
// 				std::function<void(matrix::CRS<double>&)> precon_create;
// 				std::function<void(const vector<double>& r, vector<double>& z)> precon_apply;

				using solver::solver;

				/**
				 * @brief solve Ax = b by jacobi method(lib=0: monolish)
				 * @param[in] A CRS format Matrix
				 * @param[in] x solution vector
				 * @param[in] b right hand vector
				 * @return error code (only 0 now)
				 **/
				template<typename T> int solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b);

				void precon_create(matrix::CRS<double> &A);
				void precon_create(matrix::CRS<float> &A);
				void precon_apply(const vector<float>& r, vector<float>& z);
				void precon_apply(const vector<double>& r, vector<double>& z);

// 				Jacobi(){
// 					auto create = [&precon_create](matrix::CRS<double> &A){
// 						precon_create(A);
// 					};
// 					f_precon.precon_create= std::bind(&create, ); 
// 					precon_create = std::bind(&none::precon_create<Float>, &p, std::placeholders::_1);
// 					//precon_apply = std::bind(&Jacobi::precon_apply, *this, std::placeholders::_1, std::placeholders::_2);
// 				}

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

// 				template<typename T> void precon_create(matrix::CRS<T> A)
// 				{throw std::runtime_error("error, LU cant be used as precon");}
// 				template<typename T> void precon_apply(vector<T> z, vector<T> r)
// 				{throw std::runtime_error("error, LU cant be used as precon");}
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

// 				template<typename T> void precon_create(matrix::CRS<T> A)
// 				{throw std::runtime_error("error, QR cant be used as precon");}
// 				template<typename T> void precon_apply(vector<T> z, vector<T> r)
// 				{throw std::runtime_error("error, QR cant be used as precon");}
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

// 				template<typename T> void precon_create(matrix::CRS<T> A)
// 				{throw std::runtime_error("error, Chol cant be used as precon");}
// 				template<typename T> void precon_apply(vector<T> z, vector<T> r)
// 				{throw std::runtime_error("error, Chol cant be used as precon");}
		};
	}
}
