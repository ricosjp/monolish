#/bin/bash

## Send
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Send for scalar. Performs a blocking send.
  * @param val scalar value
  * @param dst rank of dstination
  * @param tag message tag
  */"
  echo "void Send($prec val, int dst, int tag) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Send for std::vector. Performs a blocking send.
  * @param vec std::vector (size N)
  * @param dst rank of dstination
  * @param tag message tag
  */"
  echo "void Send(std::vector<$prec> &vec, int dst, int tag) const;"
done

for prec in double float; do
  echo "
  /**
  * @brief MPI_Send for monolish::vector. Performs a blocking send.
  * @param vec monolish::vector (size N)
  * @param dst rank of dstination
  * @param tag message tag
  * @warning
  * MPI functions do not support GPUs.
  * The user needs to send and receive data to and from the GPU before and after the MPI function.
  */ "
  echo "void Send(monolish::vector<$prec> &vec, int dst, int tag) const;"
done

## Recv
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Recv for scalar. Performs a blocking recv.
  * @param val scalar value
  * @param src rank of source
  * @param tag message tag
  * @return MPI status object
  */"
  echo "MPI_Status Recv($prec val, int src, int tag) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Recv for std::vector. Performs a blocking recv.
  * @param vec std::vector (size N)
  * @param src rank of source
  * @param tag message tag
  * @return MPI status object
  */"
  echo "MPI_Status Recv(std::vector<$prec> &vec, int src, int tag) const;"
done

for prec in double float; do
  echo "
  /**
  * @brief MPI_Recv for monolish::vector. Performs a blocking recv.
  * @param vec monolish::vector (size N)
  * @param src rank of source
  * @param tag message tag
  * @return MPI status object
  * @warning
  * MPI functions do not support GPUs.
  * The user needs to send and receive data to and from the GPU before and after the MPI function.
  */ "
  echo "MPI_Status Recv(monolish::vector<$prec> &vec, int src, int tag) const;"
done

