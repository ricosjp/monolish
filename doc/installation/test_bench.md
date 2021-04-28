# Tesing and Benchmarking {#test_bench}
## Testing
The program for the test can be found at `test/`.
### For CPU

```
cd test/
make cpu
make run_cpu
```

### For GPU

```
cd test/
make gpu
make run_gpu
```

## Benchmarking
The program for the test can be found at `benchmark/`.
The benchmark size is defined in `benchmark_utils.hpp`.

The results are output to each directory in tsv format.
If need to collect results in a single directory:

```
	mkdir -p result/
	cp *.tsv result/
	cp */*.tsv result/
```

### For CPU

```
cd test/
make cpu
make run_cpu
```

### For GPU

```
cd test/
make gpu
make run_gpu
```
