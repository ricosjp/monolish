#/bin/bash


## CPU Spec
CORE=`cat /proc/cpuinfo | grep processor | wc -l`
CPUS=`cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l`
MODEL=`cat /proc/cpuinfo | sed -nr '/model name/ s/.*:\s*(.*)/\1/p' | sort -u`
MEM=`lsmem | grep "online memory" | awk '{print $4}'`
GCC=`gcc --version | grep gcc`

# SIMD
SIMD=""
if [[ `grep avx512 /proc/cpuinfo` ]]; then
	SIMD="\"AVX512\""
elif [[ `grep avx2 /proc/cpuinfo` ]]; then
	SIMD="\"AVX2\""
elif [[ `grep avx /proc/cpuinfo` ]]; then
	SIMD="\"AVX\""
elif [[ `grep sse2 /proc/cpuinfo` ]]; then
	SIMD="\"SSE2\""
else
	SIMD=""
fi

# GPU
if type "nvidia-smi" > /dev/null 2>&1; then
	GPU=`nvidia-smi -L`
    if echo $GPU | grep "fail" > /dev/null 2>&1; then
        GPU="none"
    fi
else
	GPU="none"
fi

echo "\"physical_cpu\" 	 $CPUS"
echo "\"cpu\" 	 \"$MODEL\""
echo "\"cores\" 	 $CORE"
echo "\"simd\" 	 $SIMD"
echo "\"memory\" 	 \"$MEM\""
echo "\"gpu\" 	 \"$GPU\""
echo "\"gcc\" 	 \"$GCC\""
