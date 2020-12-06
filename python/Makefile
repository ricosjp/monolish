# directory where Makefile exists
MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# one level higher directory
PARENT_DIR := $(shell dirname ${MAKEFILE_DIR})
# log directory
DATA_DIR ?= $(join $(PARENT_DIR),/test/logger/logging)

all: install pylint log_cpu log_gpu

install:
	poetry install

pylint: install
	# poetry run pylint monolish_log_viewer
	poetry run pylint tests/feature/test_create_log.py
	poetry run pylint tests/unit/test_aggregate.py
	poetry run pylint tests/unit/test_html.py
	poetry run pylint tests/unit/test_read.py

log_cpu:
	poetry run monolish_log_viewer $(DATA_DIR)/monolish_test_cpu.log $(DATA_DIR)/monolish_test_cpu.html

log_gpu:
	poetry run monolish_log_viewer $(DATA_DIR)/monolish_test_gpu.log $(DATA_DIR)/monolish_test_gpu.html

test_logger:
	poetry run pytest ./tests
