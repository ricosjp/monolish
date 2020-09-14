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
    "matmul" ) 
        for format in ${FORMAT[@]}; do
            for prec in ${PREC[@]}; do
                for M in ${MAT_TEST_SIZE[@]}; do
                    for N in ${MAT_TEST_SIZE[@]}; do
                        for K in ${MAT_TEST_SIZE[@]}; do
                            $(PROFILER) ./matmul_$2.out $prec $format Dense Dense $M $N $K $3 1 || exit 1
                        done
                    done
                done
            done
        done
        ;;
    "matvec" ) 
        for format in ${FORMAT[@]}; do
            for prec in ${PREC[@]}; do
                for M in ${MAT_TEST_SIZE[@]}; do
                    for N in ${MAT_TEST_SIZE[@]}; do
                        $(PROFILER) ./matvec_$2.out $prec $format $M $N $3 1 || exit 1
                    done
                done
            done
        done
        ;;
    "mscal" ) 
        for format in ${FORMAT[@]}; do
            for prec in ${PREC[@]}; do
                for M in ${MAT_TEST_SIZE[@]}; do
                    for N in ${MAT_TEST_SIZE[@]}; do
                        .$(PROFILER) /mscal_$2.out $prec $format $M $N $3 1 || exit 1
                    done
                done
            done
        done
        ;;
    "convert" )  #COO and CRS
        for format in COO CRS; do
            for prec in ${PREC[@]}; do
                for M in ${MAT_TEST_SIZE[@]}; do
                    for N in ${MAT_TEST_SIZE[@]}; do
                        .$(PROFILER) /convert_$2.out $prec $format $M $N $3 1 || exit 1
                    done
                done
            done
        done
        ;;
esac
