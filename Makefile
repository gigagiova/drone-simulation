PYTHON := .venv/bin/python

lint:
	$(PYTHON) -m flake8 src

run:
	$(PYTHON) src/main.py
