#include "../../../include/monolish_equation.hpp"
#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef USE_GPU
	#include "cuda_runtime.h"
	#include "cusolverSp.h"
	#include "cusparse.h"
#endif


namespace monolish{


	int equation::Cholesky::cusolver_Cholesky(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

#ifdef USE_GPU

		cusolverSpHandle_t sp_handle;
		cusolverSpCreate(&sp_handle);

		cusparseMatDescr_t descrA;
		check( cusparseCreateMatDescr(&descrA) );
		check( cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL) ); 
		check( cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO) );
		check( cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT) );

		int n = A.get_row();
		int nnz = A.get_nnz();

		double* Dval = A.val.data();
		int* Dptr = A.row_ptr.data();
		int* Dind = A.col_ind.data();

		const double* Drhv = b.data();
		double* Dsol = x.data();
		int ret;

#pragma acc data copyin( Dval[0:nnz], Dptr[0:n+1], Dind[0:nnz], Drhv[0:n], Dsol[0:n] )
#pragma acc host_data use_device(Dval, Dptr, Dind, Drhv, Dsol)
  	{
		check(
				cusolverSpDcsrlsvchol(
					sp_handle,
					n,
					nnz,
					descrA,
					Dval,
					Dptr,
					Dind,
					Drhv,
					tol,
					reorder,
					Dsol,
					&singularity)
			 );
 	}
#pragma acc data copyout(Dsol[0:n])
#else
		throw std::runtime_error("error sparse Cholesky is only GPU");
#endif
		logger.func_out();
		return 0;

	}
}
