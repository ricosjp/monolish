CI_COMMIT_SHA ?= `git rev-parse HEAD`
BENCHMARK_DIR = monolish_benchmark_result
COMMIT:=https://gitlab.ritc.jp/ricos/monolish/-/commit/$(CI_COMMIT_SHA)
HOST=`hostname | sed 's/\./_/g'`

.PHONY: clone_result move_result commit_result

all:cpu

aaa:
	echo $(hostname)

clone_result:
	- rm -rf $(BENCHMARK_DIR)
ifeq ($(CI_BUILD_TOKEN),)
	git clone git@gitlab.ritc.jp:ricos/$(BENCHMARK_DIR).git
else
	git clone https://oath2:${MONOLISH_BENCHMARK_RESULT_TOKEN}@gitlab.ritc.jp/ricos/$(BENCHMARK_DIR).git
endif

move_result: clone_result
	mkdir -p $(BENCHMARK_DIR)/$(CI_COMMIT_SHA)/$(HOST)
	cp */*.tsv $(BENCHMARK_DIR)/$(CI_COMMIT_SHA)/$(HOST)
	bash ./get_spec.sh > $(BENCHMARK_DIR)/$(CI_COMMIT_SHA)/$(HOST)/spec.txt
	echo $(COMMIT) > commit_msg_tmp.txt

commit_result: clone_result move_result
	cd $(BENCHMARK_DIR) ; git add -A; git commit -m "`cat ../commit_msg_tmp.txt`"; git push origin master
	- rm commit_msg_tmp.txt
	- rm -rf $(BENCHMARK_DIR)

cpu:
	make -j -C vector_common/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_blas/ $@

gpu:
	make -j -C vector_common/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_blas/ $@

fx:
	make -j -C vector_common/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_blas/ $@

sx:
	make -j -C vector_common/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_blas/ $@

run_cpu:
	make -j -C vector_common/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_blas/ $@

run_gpu:
	make -j -C vector_common/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_blas/ $@

run_fx:
	make -j -C vector_common/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_blas/ $@

run_sx:
	make -j -C vector_common/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_blas/ $@

clean:
	make -j -C vector_common/ clean
	make -j -C vector_blas/ clean
	make -j -C matrix_common/ clean
	make -j -C matrix_blas/ clean
