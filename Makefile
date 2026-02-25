.PHONY: test test-fast test-slow test-e2e test-gates clean-workspace

test: test-fast

test-fast:
	pytest -m "not slow"

test-slow:
	pytest -m "slow"

test-e2e:
	pytest tests/e2e/test_core_workflows.py -v

test-gates:
	scripts/quality/check_test_gates.sh

clean-workspace:
	scripts/maintenance/clean_workspace.sh
