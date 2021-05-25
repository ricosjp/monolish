#!/bin/bash

## allreduce
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_SUM) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce ($prec val) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_SUM) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_sum ($prec val) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_PROD) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_prod ($prec val) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_MAX) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_max ($prec val) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_MIN) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_min ($prec val) const;"
done
