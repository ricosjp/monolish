/**
 * @author RICOS Co. Ltd.
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
					size_t rowN;
					size_t colN;
					size_t nnz;

					bool gpu_status = false; // true: sended, false: not send

				public:

					std::vector<int> row_index;
					std::vector<int> col_index;
					std::vector<Float> val;

					COO(){}

					COO(const size_t M, const size_t N, const size_t nnz, const int* row, const int* col, const Float* value){
						set_row(M);
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
					COO(const size_t M, const size_t N, const size_t nnz, const int* row, const int* col, const Float* value, const size_t origin){
						set_row(M);
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

					void set_row(const size_t M){rowN = M;};
					void set_col(const size_t N){colN = N;};
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
					size_t size() const {return get_row() > get_col() ? get_row() : get_col();}
					size_t get_row() const {return rowN;}
					size_t get_col() const {return colN;}
					size_t get_nnz() const {return nnz;}

					/**
					 * @brief matrix copy
					 * @return copied COO matrix
					 **/
					COO copy(){
                                            COO tmp(rowN, colN, nnz, row_index.data(), col_index.data(), val.data());
                                            return tmp;
					}

					std::vector<int>& get_row_ptr(){return row_index;}
					std::vector<int>& get_col_ind(){return col_index;}
					std::vector<Float>& get_val_ptr(){return val;}

                    const std::vector<int>& get_row_ptr() const {return row_index;}
                    const std::vector<int>& get_col_ind() const {return col_index;}
                    const std::vector<Float>& get_val_ptr() const {return val;}

					// Utility ///////////////////////////////////////////////////////////////////////////

                    COO& transpose() {
                        using std::swap;
                        swap(rowN, colN);
                        swap(row_index, col_index);
                        return *this;
                    }

                    void transpose(COO& B) const {
                        B.set_row(get_col());
                        B.set_col(get_row());
                        B.set_nnz(get_nnz());
                        B.row_index = get_col_ind();
                        B.col_index = get_row_ptr();
                        B.val       = get_val_ptr();
                    }

                    double get_data_size() const {
                        return 3 * get_nnz() * sizeof(Float) / 1.0e+9;
                    }

                    std::string type() const {
                        return "COO";
                    }

                    std::vector<Float> row(std::size_t i) const {
                        std::vector<Float> res(get_col(), 0);
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
                            if (get_row_ptr()[nz] == i) {
                                res[get_col_ind()[nz]] = get_val_ptr()[nz];
                            }
                        }
                        return res;
                    }

                    std::vector<Float> col(std::size_t j) const {
                        std::vector<Float> res(get_row(), 0);
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
                            if (get_col_ind()[nz] == j) {
                                res[get_row_ptr()[nz]] = get_val_ptr()[nz];
                            }
                        }
                        return res;
                    }

                    std::vector<Float> diag() const {
                        std::size_t s = get_row() > get_col() ? get_col() : get_row();
                        std::vector<Float> res(s, 0);
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
                            if (get_row_ptr()[nz] == get_col_ind()[nz]) {
                                res[get_row_ptr()[nz]] = get_val_ptr()[nz];
                            }
                        }
                        return res;
                    }

     				/////////////////////////////////////////////////////////////////////////////

					/**
					 * @brief copy matrix, It is same as copy()
					 * @param[in] filename source
					 * @return output vector
					 **/
					void operator=(const COO<Float>& mat){
						mat = copy();
					}

                    /**
                     * @brief insert element to (m, n)
                     * @param[in] size_t m row number
                     * @param[in] size_t n col number
                     * @param[in] Float val matrix value (if multiple element exists, value will be added together)
                     **/
                    void insert(size_t m, size_t n, Float val);

                 private:
                    void _q_sort(int lo, int hi);

                 public:
                    /**
                     * @brief sort COO matrix elements (and merge elements)
                     **/
                    void sort(bool merge);
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
