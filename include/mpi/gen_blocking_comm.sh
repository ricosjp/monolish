#/bin/bash

## Send
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Send for scalar. Performs a blocking send.
  * @param val scalar value
  * @param dst rank of dstination
  * @param tag message tag
  * @param gpu_sync sync gpu data. This option does not work because scalar is automatically synchronized.
  */"
  echo "void Send($prec val, int dst, int tag, bool only_cpu = false) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Send for std::vector. Performs a blocking send.
  * @param vec std::vector (size N)
  * @param dst rank of dstination
  * @param tag message tag
  * @param gpu_sync sync gpu data.This option does not work because std::vector is not support GPU.
  */"
  echo "void Send(std::vector<$prec> &vec, int dst, int tag, bool only_cpu = false) const;"
done

for prec in double float; do
  echo "
  /**
  * @brief MPI_Send for monolish::vector. Performs a blocking send.
  * @param vec monolish::vector (size N)
  * @param dst rank of dstination
  * @param tag message tag
  * @param gpu_sync sync gpu data. It receives sendvec, then performs MPI communication, and finally sends recvvec.
  * @warning
  * When "only_cpu flag is enabled" and "send data is on the GPU", data is received from the GPU, 
  * MPI communication is performed, and finally data is sent to the GPU.
  * If there is no GPU, or if there is no data on the GPU, no error will occur even if this flag is set to true.
  */ "
  echo "void Send(monolish::vector<$prec> &vec, int dst, int tag, bool only_cpu = false) const;"
done

## Recv
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Recv for scalar. Performs a blocking recv.
  * @param val scalar value
  * @param src rank of source
  * @param tag message tag
  * @param gpu_sync sync gpu data. This option does not work because scalar is automatically synchronized.
  * @return MPI status object
  */"
  echo "MPI_Status Recv($prec val, int src, int tag, bool only_cpu = false) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Recv for std::vector. Performs a blocking recv.
  * @param vec std::vector (size N)
  * @param src rank of source
  * @param tag message tag
  * @param gpu_sync sync gpu data.This option does not work because std::vector is not support GPU.
  * @return MPI status object
  */"
  echo "MPI_Status Recv(std::vector<$prec> &vec, int src, int tag, bool only_cpu = false) const;"
done

for prec in double float; do
  echo "
  /**
  * @brief MPI_Recv for monolish::vector. Performs a blocking recv.
  * @param vec monolish::vector (size N)
  * @param src rank of source
  * @param tag message tag
  * @param gpu_sync sync gpu data. It receives recvvec, then performs MPI communication, and finally recvs recvvec.
  * @return MPI status object
  * @warning
  * When "only_cpu flag is enabled" and "recv data is on the GPU", data is received from the GPU, 
  * MPI communication is performed, and finally data is sent to the GPU.
  * If there is no GPU, or if there is no data on the GPU, no error will occur even if this flag is set to true.
  */ "
  echo "MPI_Status Recv(monolish::vector<$prec> &vec, int src, int tag, bool only_cpu = false) const;"
done

