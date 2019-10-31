#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include "dmumps_c.h"
#include "mpi.h"
#include<iostream>

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654


namespace monolish{

	void equation::LU::mumps_LU(CRS_matrix<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.func_in(func);

#ifdef USE_MPI
// 		DMUMPS_STRUC_C id;
// 		MUMPS_INT n = A.get_row();
// 		MUMPS_INT8 nnz = A.get_nnz();
// 		MUMPS_INT irn = A.col_ind.data();
//
// 		std::vector<int> tmp(A.get_nnz())
// 		MUMPS_INT jcn = A.;
// 		double a[2];
// 		double rhs[2];
//
// 		MUMPS_INT myid, ierr;
//
// 		int error = 0;
// #if defined(MAIN_COMP)
// 		argv = &name;
// #endif
// 		ierr = MPI_Init(0, 0);
// 		ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
// 		/* Define A and rhs */
// 		rhs = b.data()
// 		rhs[1]=4.0;
// 		a[0]=1.0;
// 		a[1]=2.0;
//
// 		/* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
// 		id.comm_fortran=USE_COMM_WORLD;
// 		id.par=1; id.sym=0;
// 		id.job=JOB_INIT;
// 		dmumps_c(&id);
//
// 		/* Define the problem on the host */
// 		if (myid == 0) {
// 			id.n = n; id.nnz =nnz; id.irn=irn; id.jcn=jcn;
// 			id.a = a; id.rhs = rhs;
// 		}
// #define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
// 		/* No outputs */
// 		id.ICNTL(1)=-1; id.ICNTL(2)=-1; id.ICNTL(3)=-1; id.ICNTL(4)=0;
//
// 		/* Call the MUMPS package (analyse, factorization and solve). */
// 		id.job=6;
// 		dmumps_c(&id);
// 		if (id.infog[0]<0) {
// 			printf(" (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
// 					myid, id.infog[0], id.infog[1]);
// 			error = 1;
// 		}
//
// 		/* Terminate instance. */
// 		id.job=JOB_END;
// 		dmumps_c(&id);
// 		if (myid == 0) {
// 			if (!error) {
// 				printf("Solution is : (%8.2f  %8.2f)\n", rhs[0],rhs[1]);
// 			} else {
// 				printf("An error has occured, please check error code returned by MUMPS.\n");
// 			}
// 		}
// 		ierr = MPI_Finalize();
//
#endif
// 		logger.func_out();

	}


	void equation::LU::cusolver_LU(CRS_matrix<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.func_in(func);

		double* csrValA = A.val.data();
		int* csrRowPtrA = A.row_ptr.data();
		int* csrColIndA = A.col_ind.data();

		double* rhv = b.data();
		double* sol = x.data();

		int n = A.get_row();
		int nnzA = A.get_nnz();

#pragma acc update device(csrValA, csrRowPTRA, csrColIndA, rhv)
#ifdef USE_GPU
#pragma acc host_data use_device(csrValA, csrRowPtrA, csrColIndA, rhv, sol)
	{
		cusolverSpDcsrlsvlu[Host](
				int n,
				int nnzA,
				CUSPARSE_INDEX_BASE_ZERO,
				const float *csrValA,
				const int *csrRowPtrA,
				const int *csrColIndA,
				const float *rhv,
				1.0e-8,
				int reorder,
				float *sol,
				int *singularity);

	}

#endif
		logger.func_out();

	}

	void equation::LU::solve(CRS_matrix<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.func_in(func);

		if(lib == 0){
			mumps_LU(A, x, b);
		}
		else if(lib == 1){
			cusolver_LU(A, x, b);
		}

		logger.func_out();
	}

}
