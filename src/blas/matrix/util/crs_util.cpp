#include "../../../../include/common/monolish_matrix.hpp" 
#include "../../../../include/common/monolish_logger.hpp" 
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>


//kill cerr 

namespace monolish{

	template<>
		void CRS_matrix<double>::convert(COO_matrix<double> &coo){
			Logger& logger = Logger::get_instance();
			logger.func_in(func);

			//todo coo err check (only square)
			
			row = coo.get_row();
			col = coo.get_col();
			nnz = coo.get_nnz();

			val = coo.val;
			col_ind = coo.col_index;

			// todo not inplace now
			row_ptr.resize(row+1, 0.0);
			

			row_ptr[0] = 0;
			int c_row = 0;
			for (int i = 0; i < coo.get_nnz(); i++) {

				if(c_row == coo.row_index[i]){
					row_ptr[c_row+1] = i+1;
				}
				else{
					c_row = c_row + 1;
					row_ptr[c_row+1] = i+1;
				}
			}
			logger.func_out();
		}

	template<>
		void CRS_matrix<double>::output(){
			Logger& logger = Logger::get_instance();
			logger.func_in(func);

			for(int i = 0; i < row; i++){
				for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
					std::cout << i << " " << col_ind[j] << " " << val[j] << std::endl;
				}
			}

			logger.func_out();
		}
}
//
// 	template<>
// 		void COO_matrix<double>::output_mm(const char* filename){
// 			Logger& logger = Logger::get_instance();
// 			logger.func_in(func);
// 			std::ofstream out(filename);
// 			out << std::scientific;
// 			out << std::setprecision(std::numeric_limits<double>::max_digits10);
//
// 			out << (MM_BANNER " " MM_MAT " " MM_FMT " " MM_TYPE_REAL " " MM_TYPE_GENERAL) << std::endl;
// 			out << row << " " << row << " " << nnz << std::endl;
//
// 			for(int i=0; i<nnz; i++){
// 				out << row_index[i] << " " << col_index[i] << " " << val[i] << std::endl;
// 			}
// 			logger.func_out();
// 		}
//
// 	template<>
// 		void COO_matrix<double>::output(){
// 			Logger& logger = Logger::get_instance();
// 			logger.func_in(func);
// 			for(int i=0; i<nnz; i++){
// 				std::cout << row_index[i] << " " << col_index[i] << " " << val[i] << std::endl;
// 			}
// 			logger.func_out();
// 		}
//
// 	template<>
// 		double COO_matrix<double>::at(int i, int j){
// 			Logger& logger = Logger::get_instance();
// 			logger.func_in(func);
//
// 			if(i < row && j < col){
// 				throw std::runtime_error("error");
// 			}
//
// 			for(int i=0; i<nnz; i++){
// 				if( row_index[i] == i && col_index[i] == j){
// 					return val[i];
// 				}
// 			}
// 			logger.func_out();
// 			return 0.0;
// 		}
//
// 	template<>
// 		void COO_matrix<double>::set_ptr(int rN, int cN, std::vector<int> &r, std::vector<int> &c, std::vector<double> &v){
// 			col_index = c;
// 			row_index = r;
// 			val = v;
//
// 			row = rN;
// 			col = cN;
// 			nnz = r.size();
// 		}
