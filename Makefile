# indic-er
#
# Every target delegates to tasks.py, so behaviour is identical with or
# without make. On Windows without make, use `py tasks.py <target>`
# or the bundled make.bat.

PY ?= $(shell command -v py 2>/dev/null || command -v python3 2>/dev/null || echo python)

.PHONY: help setup data lint fmt test eval api demo clean

help:
	@$(PY) tasks.py help

setup:
	@$(PY) tasks.py setup

data:
	@$(PY) tasks.py data

lint:
	@$(PY) tasks.py lint

fmt:
	@$(PY) tasks.py fmt

test:
	@$(PY) tasks.py test

eval:
	@$(PY) tasks.py eval

api:
	@$(PY) tasks.py api

demo:
	@$(PY) tasks.py demo

clean:
	@$(PY) tasks.py clean
