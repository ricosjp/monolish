#include "../internal/monolish_internal.hpp"
#include "../internal/mpi/mpi_util.hpp"

namespace monolish {
namespace {

///////////////////////////
// Isend
///////////////////////////

// Scalar
template <typename T>
void Isend_core(T val, int dst, int tag, MPI_Comm comm,
                std::vector<MPI_Request> &rq) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Request request;
  rq.push_back(request);
  MPI_Isend(&val, 1, internal::mpi::get_type(val), dst, tag, comm, &rq.back());
#endif

  logger.util_out();
}

// std::vector
template <typename T>
void Isend_core(const std::vector<T> &vec, int dst, int tag, MPI_Comm comm,
                std::vector<MPI_Request> &rq) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Request request;
  rq.push_back(request);
  MPI_Isend(vec.data(), vec.size(), internal::mpi::get_type(vec.data()[0]), dst,
            tag, comm, &rq.back());
#endif

  logger.util_out();
}

// monolish::vector
template <typename T>
void Isend_core(const monolish::vector<T> &vec, int dst, int tag, MPI_Comm comm,
                std::vector<MPI_Request> &rq) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Isend(vec.data(), vec.size(), internal::mpi::get_type(vec.data()[0]), dst,
            tag, comm, &rq.back());
#endif

  logger.util_out();
}

///////////////////////////
// Irecv
///////////////////////////

// Scalar
template <typename T>
void Irecv_core(T val, int src, int tag, MPI_Comm comm,
                std::vector<MPI_Request> &rq) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Request request;
  rq.push_back(request);
  MPI_Irecv(&val, 1, internal::mpi::get_type(val), src, tag, comm, &rq.back());
#endif

  logger.util_out();
}

// std::vector
template <typename T>
void Irecv_core(std::vector<T> &vec, int src, int tag, MPI_Comm comm,
                std::vector<MPI_Request> &rq) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Request request;
  rq.push_back(request);
  MPI_Irecv(vec.data(), vec.size(), internal::mpi::get_type(vec.data()[0]), src,
            tag, comm, &rq.back());
#endif

  logger.util_out();
}

// monolish::vector
template <typename T>
void Irecv_core(monolish::vector<T> &vec, int src, int tag, MPI_Comm comm,
                std::vector<MPI_Request> &rq) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  MPI_Request request;
  rq.push_back(request);
  MPI_Irecv(vec.data(), vec.size(), internal::mpi::get_type(vec.data()[0]), src,
            tag, comm, &rq.back());
#endif

  logger.util_out();
}

void Waitall_core(std::vector<MPI_Request> &rq) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if defined MONOLISH_USE_MPI
  std::vector<MPI_Status> stat(rq.size());
  MPI_Waitall(rq.size(), rq.data(), stat.data());
  rq.clear();
#endif

  logger.util_out();
}

} // namespace
} // namespace monolish
