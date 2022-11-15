#/bin/bash

## Isend
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Isend for scalar. Performs a nonblocking send. Requests are stored internally. All requests are synchronized by Waitall().
  * @param val scalar value
  * @param dst rank of dstination
  * @param tag message tag
  * @note
  * There is not MPI_Wait() in monolish::mpi, all communication is synchronized by using Waitall() function.
  * @Warning
  * This function is not thread-safe.
  */"
  echo "void Isend($prec val, int dst, int tag);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Isend for std::vector. Performs a nonblocking send. Requests are stored internally. All requests are synchronized by Waitall().
  * @param vec std::vector (size N)
  * @param dst rank of dstination
  * @param tag message tag
  * @note
  * There is not MPI_Wait() in monolish::mpi, all communication is synchronized by using Waitall() function.
  * @Warning
  * This function is not thread-safe.
  */"
  echo "void Isend(const std::vector<$prec> &vec, int dst, int tag);"
done

for prec in double float; do
  echo "
  /**
  * @brief MPI_Isend for monolish::vector. Performs a nonblocking send. Requests are stored internally. All requests are synchronized by Waitall().
  * @param vec std::vector (size N)
  * @param dst rank of dstination
  * @param tag message tag
  * @note
  * There is not MPI_Wait() in monolish::mpi, all communication is synchronized by using Waitall() function.
  * @Warning
  * MPI functions do not support GPUs.
  * The user needs to send and receive data to and from the GPU before and after the MPI function.
  * This function is not thread-safe.
  */ "
  echo "void Isend(const monolish::vector<$prec> &vec, int dst, int tag);"
done


## Irecv
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Irecv for scalar. Performs a nonblocking recv.
  * @param val scalar value
  * @param src rank of source
  * @param tag message tag
  * @note
  * There is not MPI_Wait() in monolish::mpi, all communication is synchronized by using Waitall() function.
  * @Warning
  * This function is not thread-safe.
  */"
  echo "void Irecv($prec val, int src, int tag);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Irecv for std::vector. Performs a nonblocking recv.
  * @param vec std::vector (size N)
  * @param src rank of source
  * @param tag message tag
  * @note
  * There is not MPI_Wait() in monolish::mpi, all communication is synchronized by using Waitall() function.
  * @Warning
  * This function is not thread-safe.
  */"
  echo "void Irecv(std::vector<$prec> &vec, int src, int tag);"
done

for prec in double float; do
  echo "
  /**
  * @brief MPI_Irecv for monolish::vector. Performs a nonblocking recv.
  * @param vec monolish::vector (size N)
  * @param src rank of source
  * @param tag message tag
  * @note
  * There is not MPI_Wait() in monolish::mpi, all communication is synchronized by using Waitall() function.
  * @Warning
  * MPI functions do not support GPUs.
  * The user needs to send and receive data to and from the GPU before and after the MPI function.
  * This function is not thread-safe.
  */ "
  echo "void Irecv(monolish::vector<$prec> &vec, int src, int tag);"
done

echo "
/**
* @brief Waits for all communications to complete.
*/ "
echo "void Waitall();"
