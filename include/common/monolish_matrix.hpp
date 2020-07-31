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
#include<string>
#include<random>

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
template<typename Float> class vector;
	namespace matrix{

		/**
		 * @brief Coodinate format Matrix (need to sort)
		 */
		template<typename Float>
			class COO{
				private:

					/**
					 * @brief neet col = row now
					 */
					size_t row;
					size_t col;
					size_t nnz;

					bool gpu_status = false; // true: sended, false: not send

				public:

					std::vector<int> row_index;
					std::vector<int> col_index;
					std::vector<Float> val;

					COO(){}

					COO(const size_t N, const size_t nnz, const int* row, const int* col, const Float* value){
						set_row(N);
						set_col(N);
						set_nnz(nnz);

						row_index.resize(nnz);
						col_index.resize(nnz);
						val.resize(nnz);

						std::copy(row, row+nnz, row_index.begin());
						std::copy(col, col+nnz, col_index.begin());
						std::copy(value, value+nnz, val.begin());
					}

					// for n-origin
					COO(const size_t N, const size_t nnz, const int* row, const int* col, const Float* value, const size_t origin){
						set_row(N);
						set_col(N);
						set_nnz(nnz);

						row_index.resize(nnz);
						col_index.resize(nnz);
						val.resize(nnz);

						std::copy(row, row+nnz, row_index.begin());
						std::copy(col, col+nnz, col_index.begin());
						std::copy(value, value+nnz, val.begin());

						#pragma omp parallel for
						for(size_t i=0; i<nnz; i++){
							row_index[i] -= origin;
							col_index[i] -= origin;
						}
					}

					// communication ///////////////////////////////////////////////////////////////////////////
					/**
					 * @brief send data to GPU
					 **/
					void send(){
						throw std::runtime_error("error, GPU util of COO format is not impl. ");
					};

					/**
					 * @brief recv data from GPU
					 **/
					void recv(){
						throw std::runtime_error("error, GPU util of COO format is not impl. ");
					};

					/**
					 * @brief free data on GPU
					 **/
					void device_free(){
					};

					/**
					 * @brief false; // true: sended, false: not send
					 * @return true is sended.
					 * **/
					bool get_device_mem_stat() const{ return gpu_status; }

					/**
					 * @brief; free gpu mem.
					 * **/
					~ COO(){
						if(get_device_mem_stat()){
							device_free();
						}
					}

					// I/O ///////////////////////////////////////////////////////////////////////////

					void set_row(const size_t M){row = M;};
					void set_col(const size_t N){col = N;};
					void set_nnz(const size_t NNZ){nnz = NNZ;};

					void input_mm(const char* filename);

					COO(const char* filename){
						input_mm(filename);
					}

					/**
					 * @brief print all elements to standart I/O
					 **/
					void print_all();
					/**
					 * @brief print all elements to file
					 * @param[in] filename output filename
					 **/
					void print_all(std::string filename);
					Float at(size_t i, size_t j);
					//void insert(size_t i, size_t j, Float value);

					void set_ptr(size_t rN, size_t cN, std::vector<int> &r, std::vector<int> &c, std::vector<Float> &v);

					//not logging, only square
					size_t size() const {return row > col ? row : col;}
					size_t get_row(){return row;}
					size_t get_col(){return col;}
					size_t get_nnz(){return nnz;}

					/**
					 * @brief matrix copy
					 * @return copied COO matrix
					 **/
					COO copy(){
						COO tmp(row, nnz, row_index.data(), col_index.data(), val.data());
						return tmp;
					}

					std::vector<int>& get_row_ptr(){return row_index;}
					std::vector<int>& get_col_ind(){return col_index;}
					std::vector<Float>& get_val_ptr(){return val;}

     				/////////////////////////////////////////////////////////////////////////////

					/**
					 * @brief copy matrix, It is same as copy()
					 * @param[in] filename source
					 * @return output vector
					 **/
					void operator=(const COO<Float>& mat){
						mat = copy();
					}

			};

		/**
		 * @brief CRS format Matrix
		 */
		template<typename Float>
			class CRS{
				private:
					/**
					 * @brief neet col = row now
					 */
					size_t row;
					size_t col;
					size_t nnz;

					bool gpu_flag = false; // not impl
					bool gpu_status = false; // true: sended, false: not send

				public:
					std::vector<Float> val;
					std::vector<int> col_ind;
					std::vector<int> row_ptr;

					void convert(COO<Float> &coo);

					CRS(){}

					CRS(COO<Float> &coo){
						convert(coo);
					}

					CRS(const CRS<Float> &mat);

					void output();

					size_t size() const {return row > col ? row : col;}
					size_t get_row() const{return row;}
					size_t get_col() const{return col;}
					size_t get_nnz() const{return nnz;}

					// communication ///////////////////////////////////////////////////////////////////////////
					/**
					 * @brief send data to GPU
					 **/
					void send();

					/**
					 * @brief recv and free data from GPU
					 **/
					void recv();

					/**
					 * @brief recv data from GPU (w/o free)
					 **/
					void nonfree_recv();

					/**
					 * @brief free data on GPU
					 **/
					void device_free();

					/**
					 * @brief false; // true: sended, false: not send
					 * @return true is sended.
					 * **/
					bool get_device_mem_stat() const{ return gpu_status; }

					/**
					 * @brief; free gpu mem.
					 * **/
					~ CRS(){
						if(get_device_mem_stat()){
							device_free();
						}
					}

     				/////////////////////////////////////////////////////////////////////////////
					void get_diag(vector<Float>& vec);
					void get_row(const size_t r, vector<Float>& vec);
					void get_col(const size_t c, vector<Float>& vec);

     				/////////////////////////////////////////////////////////////////////////////

					/**
					 * @brief matrix copy
					 * @return copied CRS matrix
					 **/
					CRS copy();

					/**
					 * @brief copy matrix, It is same as copy()
					 * @return output matrix
					 **/
					void operator=(const CRS<Float>& mat);

					//mat - vec
					vector<Float> operator*(vector<Float>& vec);

					//mat - scalar
					CRS<Float> operator*(const Float value);
			};
	}
}
