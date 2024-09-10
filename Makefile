.PHONY: build update install lint unit-test integration-test tests_lints run-streamlit clean

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

export PYTHON_VERSION=3.8.12

build:
	poetry build

update:
	poetry update

install:
	poetry install

lint:
	$(ROOT_DIR)/dev/run_all_lints.sh

unit-test:
	$(ROOT_DIR)/dev/run_all_unit_tests.sh

integration-test:
	poetry run pytest tests/integration/experiment

tests_lints:
	$(MAKE) lint
	$(MAKE) unit-test
	$(MAKE) integration-test

run-mintlify:
	cd $(ROOT_DIR)/docs; mintlify dev

clean:
	$(ROOT_DIR)/dev/clean_dev_env.sh