#include "../../include/common/monolish_matrix.hpp" 
#include "../../include/common/monolish_vector.hpp" 
#include "../../include/common/monolish_logger.hpp" 
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>


//kill cerr 

namespace monolish{
	namespace matrix{

		template<typename T>
			void CRS<T>::convert(COO<T> &coo){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);

				//todo coo err check (only square)

				row = coo.get_row();
				col = coo.get_col();
				nnz = coo.get_nnz();

				val = coo.val;
				col_ind = coo.col_index;

				// todo not inplace now
				row_ptr.resize(row+1, 0.0);


				row_ptr[0] = 0;
				size_t c_row = 0;
				for (size_t i = 0; i < coo.get_nnz(); i++) {

					if((int)c_row == coo.row_index[i]){
						row_ptr[c_row+1] = i+1;
					}
					else{
						c_row = c_row + 1;
						row_ptr[c_row+1] = i+1;
					}
				}
				logger.util_out();
			}
		template void CRS<double>::convert(COO<double> &coo);
		template void CRS<float>::convert(COO<float> &coo);

		template<typename T>
			void CRS<T>::output(){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);

				for(size_t i = 0; i < row; i++){
					for(size_t j = (size_t)row_ptr[i]; j < (size_t)row_ptr[i+1]; j++){
						std::cout << i+1 << " " << col_ind[j]+1 << " " << val[j] << std::endl;
					}
				}

				logger.util_out();
			}
		template void CRS<double>::output();
		template void CRS<float>::output();
	}
}
