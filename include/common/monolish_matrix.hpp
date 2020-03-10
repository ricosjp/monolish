/**
 * @autor RICOS Co. Ltd.
 * @file monolish_vector.h
 * @brief declare vector class
 * @date 2019
 **/

#pragma once
#include<omp.h>
#include<exception>
#include<stdexcept>
#include<vector>

#define MM_BANNER "%%MatrixMarket"
#define MM_MAT "matrix"
#define MM_VEC "vector"
#define MM_FMT "coordinate"
#define MM_TYPE_REAL "real"
#define MM_TYPE_GENERAL "general"
#define MM_TYPE_SYMM "symmetric"

#if defined USE_MPI
#include<mpi.h>
#endif

namespace monolish{
	namespace matrix{

		/**
		 * @brief Coodinate format Matrix (need to sort)
		 */
		template<typename Float>
			class COO{
				private:
					size_t row;

					/**
					 * @brief neet col = row now
					 */
					size_t col;
					size_t nnz;

					bool gpu_flag = false; // not impl

					void set_rowN(const size_t N){row = N;};
					void set_colN(const size_t N){col = N;};
					void set_nnzN(const size_t N){nnz = N;};

				public:

					std::vector<int> row_index;
					std::vector<int> col_index;
					std::vector<Float> val;

					COO(){}

					COO(const int N, const int nnz, const int* row, const int* col, const double* value){
						set_rowN(N);
						set_colN(N);
						set_nnzN(nnz);

						row_index.resize(nnz);
						col_index.resize(nnz);
						val.resize(nnz);

						std::copy(row, row+nnz, row_index.begin());
						std::copy(col, col+nnz, col_index.begin());
						std::copy(value, value+nnz, val.begin());
					}

					void input_mm(const char* filename);

					COO(const char* filename){
						input_mm(filename);
					}

					void output_mm(const char* filename);
					void output();
					double at(int i, int j);
					//void insert(int i, int j, double value);

					void set_ptr(int rN, int cN, std::vector<int> &r, std::vector<int> &c, std::vector<double> &v);

					//not logging, only square
					int get_row(){return row;}
					int get_col(){return col;}
					int get_nnz(){return nnz;}

					std::vector<int>& get_row_p(){return row_index;}
					std::vector<int>& get_col_p(){return col_index;}
					std::vector<Float>& get_val_p(){return val;}
			};

		/**
		 * @brief CRS format Matrix
		 */
		template<typename Float>
			class CRS{
				private:
					int row;

					/**
					 * @brief neet col = row now
					 */
					int col;
					int nnz;

					bool gpu_flag = false; // not impl
					void convert(COO<double> &coo);

				public:
					std::vector<Float> val;
					std::vector<int> col_ind;
					std::vector<int> row_ptr;

					CRS(){}
					CRS(COO<double> &coo){
						convert(coo);
					}

					void output();
					// 				void at(int i, int j);
					// 				void set_ptr(std::vector<int> &r, std::vector<int> &c, std::vector<double> &v);

					//not logging
					int get_row(){return row;}
					int get_col(){return col;}
					int get_nnz(){return nnz;}
			};
	}
}
