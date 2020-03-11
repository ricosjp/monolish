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

					std::vector<size_t> row_index;
					std::vector<size_t> col_index;
					std::vector<Float> val;

					COO(){}

					COO(const size_t N, const size_t nnz, const size_t* row, const size_t* col, const double* value){
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
					double at(size_t i, size_t j);
					//void insert(size_t i, size_t j, double value);

					void set_ptr(size_t rN, size_t cN, std::vector<size_t> &r, std::vector<size_t> &c, std::vector<double> &v);

					//not logging, only square
					size_t get_row(){return row;}
					size_t get_col(){return col;}
					size_t get_nnz(){return nnz;}

					std::vector<size_t>& get_row_p(){return row_index;}
					std::vector<size_t>& get_col_p(){return col_index;}
					std::vector<Float>& get_val_p(){return val;}
			};

		/**
		 * @brief CRS format Matrix
		 */
		template<typename Float>
			class CRS{
				private:
					size_t row;

					/**
					 * @brief neet col = row now
					 */
					size_t col;
					size_t nnz;

					bool gpu_flag = false; // not impl
					void convert(COO<double> &coo);

				public:
					std::vector<Float> val;
					std::vector<size_t> col_ind;
					std::vector<size_t> row_ptr;

					CRS(){}
					CRS(COO<double> &coo){
						convert(coo);
					}

					void output();
					// 				void at(size_t i, size_t j);
					// 				void set_ptr(std::vector<size_t> &r, std::vector<size_t> &c, std::vector<double> &v);

					//not logging
					size_t get_row(){return row;}
					size_t get_col(){return col;}
					size_t get_nnz(){return nnz;}
			};
	}
}
