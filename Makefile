UV ?= uv
PYTHON ?= python3
SERVER_NAME ?= s3vectors
INSTALL_DIR ?= $(HOME)/.mcp/servers/$(SERVER_NAME)

.PHONY: install uninstall test lint fmt

install:
	$(UV) tool install --force --python 3.11 .
	$(PYTHON) scripts/mcp_install.py --install-dir "$(INSTALL_DIR)" --server-name "$(SERVER_NAME)"

uninstall:
	rm -rf "$(INSTALL_DIR)"

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check

fmt:
	$(UV) run ruff format

