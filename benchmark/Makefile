CI_COMMIT_SHA ?= `git rev-parse HEAD`
CI_PIPELLINE_ID ?= 0

BENCHMARK_DIR:=monolish_benchmark_result
VERSION_DIR_NAME=$(CI_COMMIT_SHA)

COMMIT_MSG:=https://github.com/ricosjp/monolish/commit/$(CI_COMMIT_SHA)

.PHONY: clone_result move_result commit_result

all:cpu gpu

clone_result:
	- rm -rf $(BENCHMARK_DIR)
ifeq ($(CI_BUILD_TOKEN),)
	git clone git@gitlab.ritc.jp:ricos/$(BENCHMARK_DIR).git
else
	git clone https://oath2:${MONOLISH_BENCHMARK_RESULT_TOKEN}@gitlab.ritc.jp/ricos/$(BENCHMARK_DIR).git
endif

move_result: clone_result
	mkdir -p $(BENCHMARK_DIR)/data/$(VERSION_DIR_NAME)/
	cp *.tsv $(BENCHMARK_DIR)/data/$(VERSION_DIR_NAME)/
	cp */*.tsv $(BENCHMARK_DIR)/data/$(VERSION_DIR_NAME)/
	echo $(COMMIT_MSG) > commit_msg_tmp.txt

commit_result: clone_result move_result
	cd $(BENCHMARK_DIR) ; git add -A; git commit -m "`cat ../commit_msg_tmp.txt`"; git push origin master
	- rm commit_msg_tmp.txt
	- rm -rf $(BENCHMARK_DIR)

get_spec_intel_mkl:
	bash ./get_spec.sh > spec_intel_mkl.tsv

get_spec_intel_oss:
	bash ./get_spec.sh > spec_intel_oss.tsv

get_spec_amd_oss:
	bash ./get_spec.sh > spec_amd_oss.tsv

get_spec_nvidia:
	bash ./get_spec.sh > spec_nvidia.tsv

get_ci_info:
	bash ./get_version_info.sh > version_info.tsv


cpu: intel_mkl
gpu: run_nvidia
run_cpu: run_intel_mkl
run_gpu: run_nvidia

intel_mkl:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

intel_oss:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

amd_oss:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

nvidia:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

a64fx:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

sxat:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

run_intel_mkl:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

run_intel_oss:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

run_amd_oss:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

run_nvidia:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

run_a64fx:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

run_sxat:
	make -j -C vector_vml/ $@
	make -j -C vector_blas/ $@
	make -j -C matrix_common/ $@
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ $@

clean:
	make -j -C vector_vml/ clean
	make -j -C vector_blas/ clean
	make -j -C matrix_common/ clean
	make -j -C matrix_vml/ $@
	make -j -C matrix_blas/ clean
	- rm -rf $(BENCHMARK_DIR)
	- rm commit_msg_tmp.txt
