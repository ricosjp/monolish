#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T>
void LinearOperator<T>::operator=(const LinearOperator<T> &linearoperator) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  rowN = linearoperator.get_row();
  colN = linearoperator.get_col();

  if (linearoperator.get_matvec_init_flag()) {
    set_matvec(linearoperator.get_matvec());
  }
  if (linearoperator.get_rmatvec_init_flag()) {
    set_rmatvec(linearoperator.get_rmatvec());
  }

  logger.util_out();
}

template void
LinearOperator<double>::operator=(const LinearOperator<double> &linearoperator);
template void
LinearOperator<float>::operator=(const LinearOperator<float> &linearoperator);

} // namespace matrix
} // namespace monolish
