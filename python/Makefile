# directory where Makefile exists
MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# one level higher directory
PARENT_DIR := $(shell dirname ${MAKEFILE_DIR})
# log directory
DATA_DIR ?= $(join $(PARENT_DIR),/test/logger/logging)

all: log_cpu log_gpu

log_cpu:
	poetry run monolish_log_viewer $(DATA_DIR)/monolish_test_cpu.log $(DATA_DIR)/monolish_test_cpu.html

log_gpu:
	poetry run monolish_log_viewer $(DATA_DIR)/monolish_test_gpu.log $(DATA_DIR)/monolish_test_gpu.html

test_logger:
	poetry run pytest ./tests/unit/utils/test_html.py -s
	poetry run pytest ./tests/unit/utils/test_read.py -s
	poetry run pytest ./tests/unit/libs/test_grouping.py -s
	poetry run pytest ./tests/unit/libs/test_aggregate.py -s
	poetry run pytest ./tests/feature/test_create_log.py -s
