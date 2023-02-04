#!/bin/bash

if [ $# != 2 ]; then
    echo "error!, \$1: func, \$2: arch."
    exit 1
fi

FORMAT=(Dense CRS)
PREC=(double float)
TEST_A=$(($RANDOM%50+50)) #50~100
TEST_B=$(($RANDOM%50+50)) #50~100
MAT_TEST_SIZE=($TEST_A $TEST_B)

PROFILER="valgrind --leak-check=full"

case $1 in
    "matrix_common" ) 
        $PROFILER ./$1_$2.out $prec $format $M $N 1 || exit 1
        ;;

    "matrix_blas" ) 
        for M in ${MAT_TEST_SIZE[@]}; do
            for N in ${MAT_TEST_SIZE[@]}; do
                for K in ${MAT_TEST_SIZE[@]}; do
                    $PROFILER ./$1_$2.out $M $N $K || exit 1
                done
            done
        done
        ;;

    "matrix_convert" ) 
        for M in ${MAT_TEST_SIZE[@]}; do
            for N in ${MAT_TEST_SIZE[@]}; do
                $PROFILER ./$1_$2.out $M $N || exit 1
            done
        done
        ;;

    "matrix_compare" ) 
        $PROFILER ./$1_$2.out $prec $format $M $N 1 || exit 1
        ;;

    "matrix_vml" ) 
        for M in ${MAT_TEST_SIZE[@]}; do
            for N in ${MAT_TEST_SIZE[@]}; do
                for K in ${MAT_TEST_SIZE[@]}; do
                    $PROFILER ./$1_$2.out $M $N $K || exit 1
                done
            done
        done
        ;;

    "matrix_transpose" ) 
        for M in ${MAT_TEST_SIZE[@]}; do
            for N in ${MAT_TEST_SIZE[@]}; do
                $PROFILER ./$1_$2.out $M $N || exit 1
            done
        done
        ;;

    "matrix_ subvec_op" ) 
        for M in ${MAT_TEST_SIZE[@]}; do
            for N in ${MAT_TEST_SIZE[@]}; do
                $PROFILER ./$1_$2.out $M $N || exit 1
            done
        done
        ;;
esac
