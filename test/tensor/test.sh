#!/bin/bash

if [ $# != 2 ]; then
  echo "error!, \$1: func, \$2: arch."
  exit 1
fi

FORMAT=(Dense)
PREC=(double float)
TEST_A=$(($RANDOM%8+8)) #8~16
TEST_B=$(($RANDOM%8+8)) #8~16
MAT_TEST_SIZE=($TEST_A $TEST_B)

case $1 in
  "tensor_common" )
    $PROFILER ./$1_$2.out $prec $format $M $N $L 1 || exit 1
    ;;

  "tensor_compare" )
    $PROFILER ./$1_$2.out $prec $format $M $N 1 || exit 1
    ;;

  "tensor_vml" ) 
      for M in ${MAT_TEST_SIZE[@]}; do
          for N in ${MAT_TEST_SIZE[@]}; do
              for K in ${MAT_TEST_SIZE[@]}; do
                  $PROFILER ./$1_$2.out $M $N $K || exit 1
              done
          done
      done
      ;;

  "tensor_blas" )
    for M in ${MAT_TEST_SIZE[@]}; do
      for N in ${MAT_TEST_SIZE[@]}; do
        for K in ${MAT_TEST_SIZE[@]}; do
          for L in ${MAT_TEST_SIZE[@]}; do
            for J in ${MAT_TEST_SIZE[@]}; do
              $PROFILER ./$1_$2.out $M $N $K $L $J || exit 1
            done
          done
        done
      done
    done
    ;;

  "tensor_convert" )
    for M in ${MAT_TEST_SIZE[@]}; do
      for N in ${MAT_TEST_SIZE[@]}; do
        for L in ${MAT_TEST_SIZE[@]}; do
          $PROFILER ./$1_$2.out $M $N $L || exit 1
        done
      done
    done
    ;;

esac

