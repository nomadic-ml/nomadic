.PHONY: build update install setup_dev_environment lint unit_test integration_test tests_lints run_mintlify clean

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

export PYTHON_VERSION=3.11.9

build:
	poetry build

update:
	poetry update

install:
	poetry install

setup_dev_environment:
	$(ROOT_DIR)/dev/build_dev_env.sh

lint:
	$(ROOT_DIR)/dev/run_all_lints.sh

unit_test:
	$(ROOT_DIR)/dev/run_all_unit_tests.sh

integration_test:
	poetry run pytest tests/integration/experiment

tests_lints:
	$(MAKE) lint
	$(MAKE) unit_test
	$(MAKE) integration_test

run_mintlify:
	cd $(ROOT_DIR)/docs; mintlify dev

clean:
	$(ROOT_DIR)/dev/clean_dev_env.sh
