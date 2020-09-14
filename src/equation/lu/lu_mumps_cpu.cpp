#include "../../../include/monolish_blas.hpp"
#include "../../../include/monolish_equation.hpp"
#include <iostream>

//	#include "dmumps_c.h"
//	#include "mpi.h"

#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

namespace monolish {

// mumps is choushi warui..
template <>
int equation::LU<double>::mumps_LU(matrix::CRS<double> &A, vector<double> &x,
                                   vector<double> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (1) {
    throw std::runtime_error("error sparse LU on CPU does not impl.");
  }

  // 		DMUMPS_STRUC_C id;
  // 		MUMPS_INT n = A.get_row();
  // 		MUMPS_INT8 nnz = A.get_nnz();
  //
  // 		// covert mumps format (CRS -> 1-origin COO)
  // 		std::vector<int>tmp_row(nnz);
  // 		std::vector<int>tmp_col(nnz);
  // 		for(int i=0; i<n; i++){
  // 			for(int j = A.row_ptr[i]; j < A.row_ptr[i+1]; j++){
  // 				tmp_row[j] = i+1;
  // 				tmp_row[j] = A.col_ind[j]+1;
  // 			}
  // 		}
  //   		MUMPS_INT* irn = A.row_ptr.data();
  //   		MUMPS_INT* jcn = A.col_ind.data();
  //
  // 		double* a = A.val.data();
  // 		double* rhs = b.data();
  //
  // 		MUMPS_INT myid, ierr;
  // 		int* dummy;
  // 		char*** dummyc;
  //
  // 		int error = 0;
  // 		#if defined(MAIN_COMP)
  // 			argv = &name;
  // 		#endif
  // 		ierr = MPI_Init(dummy, dummyc);
  // 		ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
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
  //
  // #define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
  // 		/* No outputs */
  // 		id.ICNTL(1)=-1; id.ICNTL(2)=-1; id.ICNTL(3)=-1; id.ICNTL(4)=0;
  //
  // 		/* Call the MUMPS package (analyse, factorization and solve). */
  // 		id.job=6;
  // 		dmumps_c(&id);
  // 		if (id.infog[0]<0) {
  // 			printf(" (PROC %d) ERROR RETURN: \tINFOG(1)=
  // %d\n\t\t\t\tINFOG(2)=
  // %d\n", 					myid, id.infog[0], id.infog[1]);
  // error = 1;
  // 		}
  //
  // 		/* Terminate instance. */
  // 		id.job=JOB_END;
  // 		dmumps_c(&id);
  // 		if (myid == 0) {
  // 			if (!error) {
  // 				printf("Solution is : (%8.2f  %8.2f)\n",
  // rhs[0],rhs[1]); 			} else {
  // printf("An error has occured, please check error code returned by
  // MUMPS.\n");
  // 			}
  // 		}
  // 		ierr = MPI_Finalize();

  logger.func_out();
  return 0;
}
} // namespace monolish
