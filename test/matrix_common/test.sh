#!/bin/bash

if [ $# != 3 ]; then
    echo "error!, \$1: func, \$2: arch., \$3: iter."
    exit 1
fi

FORMAT=(Dense CRS)
PREC=(double float)
TEST_A=$(($RANDOM%100+100)) #100~200
TEST_B=$(($RANDOM%100+100)) #100~200
MAT_TEST_SIZE=($TEST_A $TEST_B)

case $1 in
    "convert" ) 
        for format in ${FORMAT[@]}; do
            for prec in ${PREC[@]}; do
                for M in ${MAT_TEST_SIZE[@]}; do
                    for N in ${MAT_TEST_SIZE[@]}; do
                        echo start $1 $2 $prec $format $M $N $3
                        $PROFILER ./$1_$2.out $prec $format $M $N $3 1 || exit 1
                    done
                done
            done
        done
        ;;
    "mv_mul" ) 
        for format in ${FORMAT[@]}; do
            for prec in ${PREC[@]}; do
                for M in ${MAT_TEST_SIZE[@]}; do
                    for N in ${MAT_TEST_SIZE[@]}; do
                        echo start $1 $2 $prec $format $M $N $3
                        $PROFILER ./$1_$2.out $prec $format $M $N $3 1 || exit 1
                    done
                done
            done
        done
        ;;
    "transpose" ) 
        for format in Dense; do # only Dense
            for prec in ${PREC[@]}; do
                for M in ${MAT_TEST_SIZE[@]}; do
                    for N in ${MAT_TEST_SIZE[@]}; do
                        echo start $1 $2 $prec $format $M $N $3
                        $PROFILER ./$1_$2.out $prec $format $M $N $3 1 || exit 1
                    done
                done
            done
        done
        ;;
    "matrix_common" ) 
        echo start $1 $prec $format $M $N $3
        $PROFILER ./$1_$2.out $prec $format $M $N $3 1 || exit 1
        ;;
esac
