## ─────────────────────────────────────────────────────────────────────────────
## Washington State EV Analysis — Makefile
## Works on: Windows (cmd.exe, PowerShell, Git Bash), Linux, macOS
## Uses inline Python so no shell-branching is needed.
## ─────────────────────────────────────────────────────────────────────────────

PYTHON ?= python
PIP    ?= pip

.PHONY: help install lint format test coverage clean run-app download-geojson

## help: list available targets
help:
	@$(PYTHON) -c "import re,sys; \
	[print(f'  {m.group(1):<18} {m.group(2)}') \
	 for line in open('Makefile') \
	 for m in [re.match(r'^## (\S.*?):\s+(.*)', line)] if m]"

## install: install all dependencies
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

## lint: run ruff linter
lint:
	ruff check src/ tests/ app/

## format: auto-format with ruff
format:
	ruff format src/ tests/ app/

## test: run unit tests
test:
	pytest tests/ -v

## coverage: run tests with coverage report
coverage:
	pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=35

## download-geojson: bundle US counties GeoJSON locally (avoids network fetch at runtime)
download-geojson:
	curl -fsSL -o data/us_counties.json \
		https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json
	@echo "GeoJSON saved to data/us_counties.json"

## run-app: launch the Streamlit dashboard
run-app:
	streamlit run app/streamlit_app.py

## clean: remove compiled Python files and cache dirs
clean:
	$(PYTHON) -c "\
import shutil, pathlib; \
[shutil.rmtree(p, ignore_errors=True) \
 for p in pathlib.Path('.').rglob('__pycache__')]; \
[shutil.rmtree(p, ignore_errors=True) \
 for p in pathlib.Path('.').rglob('.pytest_cache')]; \
[shutil.rmtree(p, ignore_errors=True) \
 for p in pathlib.Path('.').rglob('*.egg-info')]; \
[p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]; \
print('Cleaned.')"