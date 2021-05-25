#!/bin/bash

## allreduce
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_SUM) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce const($prec val);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_SUM) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_sum const($prec val);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_PROD) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_prod const($prec val);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_MAX) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_max const($prec val);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_MIN) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_min const($prec val);"
done
