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
	poetry run DATA_DIR $(DATA_DIR)/monolish_test_gpu.log $(DATA_DIR)/monolish_test_gpu.html
