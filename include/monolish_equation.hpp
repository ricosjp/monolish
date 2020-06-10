#pragma once
#include<vector>

#if defined USE_MPI
#include<mpi.h>
#endif

#include"common/monolish_common.hpp"
#include<functional>

namespace monolish::equation{

    template<typename Float> class precondition;

    template<typename Float>
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

            Float get_residual(vector<Float>& x);

            precondition<Float> precond;

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
            template<class PRECOND> void set_precond_create(PRECOND& p);

            /**
             * @brief set precondition apply fucntion
             * @param[in] f function 
             **/
            template<class PRECOND>void set_precond_apply(PRECOND& p);

            /**
             * @brief set library option (to change library, monolish, cusolver, etc.)
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
     * @brief none solver class
     */
    template<typename Float>
        class precondition{
            private:
            public:
                vector<Float> M;
                monolish::matrix::CRS<Float> tmpA;

                std::function<void(matrix::CRS<Float>&)> precond_create;
                std::function<void(const vector<Float>& r, vector<Float>& z)> precond_apply;

                std::function<void(void)> get_precond();

                void set_precond_data(vector<Float>& m){M=m;};
                vector<Float> get_precond_data() {return M;};

                precondition(){};
        };


    /**
     * @brief none solver class
     */
    template <typename Float>
        class none : public solver<Float>{
            public:
                void precond_create(matrix::CRS<Float>& A);
                void precond_apply(const vector<Float>& r, vector<Float>& z);
                int solve(matrix::CRS<Float> &A, vector<Float> &x, vector<Float> &b);
        };

    /**
     * @brief CG solver class
     */
    template <typename Float>
        class CG : public solver<Float>{
            private:
                int monolish_CG(matrix::CRS<Float> &A, vector<Float> &x, vector<Float> &b);
            public:
                /**
                 * @brief solve Ax = b by CG method(lib=0: monolish)
                 * @param[in] A CRS format Matrix
                 * @param[in] x solution vector
                 * @param[in] b right hand vector
                 * @return error code (only 0 now)
                 **/
                int solve(matrix::CRS<Float> &A, vector<Float> &x, vector<Float> &b);
                void precond_create(matrix::CRS<Float> &A){
                    throw std::runtime_error("this precond. is not impl.");
                }
                void precond_apply(const vector<Float>& r, vector<Float>& z){
                    throw std::runtime_error("this precond. is not impl.");
                }

        };

    //jacobi////////////////////////////////
    template <typename Float>
    class Jacobi : public solver<Float>{
        private:
            int monolish_Jacobi(matrix::CRS<Float> &A, vector<Float> &x, vector<Float> &b);
        public:
            /**
             * @brief solve Ax = b by jacobi method(lib=0: monolish)
             * @param[in] A CRS format Matrix
             * @param[in] x solution vector
             * @param[in] b right hand vector
             * @return error code (only 0 now)
             **/
            int solve(matrix::CRS<Float> &A, vector<Float> &x, vector<Float> &b);
            void precond_create(matrix::CRS<Float> &A);
            void precond_apply(const vector<Float>& r, vector<Float>& z);
    };


    /**
     * @brief LU solver class (does not impl. now)
     */
    template <typename Float>
    class LU : public solver<Float>{
        private:
            int lib = 1; // lib is 1
            int mumps_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
            int cusolver_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
            int singularity;
            int reorder = 3;

        public:
            void set_reorder(int r){ reorder = r; }
            int get_sigularity(){ return singularity; }
            int solve(matrix::CRS<Float> &A, vector<Float> &x, vector<Float> &b);
            void precond_create(matrix::CRS<Float> &A){
			    throw std::runtime_error("this precond. is not impl.");
            }
            void precond_apply(const vector<Float>& r, vector<Float>& z){
			    throw std::runtime_error("this precond. is not impl.");
            }
    };

    /**
     * @brief QR solver class (GPU only now). can use set_tol(), get_til(),
     * set_reorder(), get_singularity(). default reorder algorithm is csrmetisnd
     */
    template <typename Float>
    class QR : public solver<Float>{
        private:
            int lib = 1; // lib is 1
            int cusolver_QR(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
            int cusolver_QR(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
            int singularity;
            int reorder = 3;

        public:

            /**
             * @brief 0: no ordering 1: symrcm, 2: symamd, 3: csrmetisnd is used to reduce zero fill-in.
             */
            void set_reorder(int r){ reorder = r; }

            /**
             * @brief -1 if A is symmetric postive definite.
             */
            int get_sigularity(){ return singularity; }

            /**
             * @brief solve Ax=b
             */
            int solve(matrix::CRS<Float> &A, vector<Float> &x, vector<Float> &b);
            void precond_create(matrix::CRS<Float> &A){
			    throw std::runtime_error("this precond. is not impl.");
            }
            void precond_apply(const vector<Float>& r, vector<Float>& z){
			    throw std::runtime_error("this precond. is not impl.");
            }
    };

    /**
     * @brief Cholesky solver class (GPU only now). can use set_tol(), get_til(),
     * set_reorder(), get_singularity(). default reorder algorithm is csrmetisnd
     */
    template <typename Float>
    class Cholesky : public solver<Float>{
        private:
            int lib = 1; // lib is 1
            int cusolver_Cholesky(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
            int cusolver_Cholesky(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
            int singularity;
            int reorder = 3;

        public:

            /**
             * @brief 0: no ordering 1: symrcm, 2: symamd, 3: csrmetisnd is used to reduce zero fill-in.
             */
            void set_reorder(int r){ reorder = r; }

            /**
             * @brief -1 if A is symmetric postive definite.
             */
            int get_sigularity(){ return singularity; }

            /**
             * @brief solve Ax=b
             */
            int solve(matrix::CRS<Float> &A, vector<Float> &x, vector<Float> &b);

            void precond_create(matrix::CRS<Float> &A){
			    throw std::runtime_error("this precond. is not impl.");
            }
            void precond_apply(const vector<Float>& r, vector<Float>& z){
			    throw std::runtime_error("this precond. is not impl.");
            }
    };
}
