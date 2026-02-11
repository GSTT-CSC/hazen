#############
# Variables #
#############

VENV_CMD = uv run
SRC_DIR = hazenlib
TEST_DIR = tests
DOCS_DIR = docs

ACR_DATA = tests/data/acr/GE_Artist_1.5T_T1

ACR_DATA_T1 = tests/data/acr/GE_Artist_1.5T_T1
ACR_DATA_T2 = tests/data/acr/GE_Artist_1.5T_T2
ACR_DATA_SL = tests/data/acr/GE_Signa_1.5T_Sagittal_Localizer

##################
# Default Target #
##################

.DEFAULT_GOAL := help

###############
# Help Target #
###############

# Collects all commands that follow the format: cmd: ... ## Description

.PHONY: help
help:
	@echo "Hazen Development Makefile"
	@echo "======================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

################
# Installation #
################

.PHONY: install
install: ## Install dependencies and sync virtualenv
	@echo "Installing dependencies..."
	uv sync

.PHONY: lock
lock: ## Update lockfile with latest dependencies
	@echo "Updating dependencies..."
	uv lock

###########
# Testing #
###########

.PHONY: test
test: ## Run pytest with coverage
	@echo "Running tests..."
	$(VENV_CMD) pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

.PHONY: test-fast
test-fast: ## Run tests without coverage (faster)
	@echo "Running tests (fast mode)..."
	$(VENV_CMD) pytest $(TEST_DIR) -v

#################
# Type Checking #
#################

.PHONY: type-check
type-check: mypy ty ## Run all type checkers

.PHONY: mypy
mypy: ## Run mypy type checker
	@echo "Running mypy..."
	$(VENV_CMD) mypy $(SRC_DIR) $(TEST_DIR)

.PHONY: ty
ty: ## Run ty type checker
	@echo "Running ty..."
	$(VENV_CMD) ty $(SRC_DIR)

##################
# Linting/Format #
##################

.PHONY: lint
lint: ## Run ruff linter
	@echo "Running linter..."
	$(VENV_CMD) ruff check $(SRC_DIR) $(TEST_DIR)

.PHONY: format
format: ## Format code with ruff
	@echo "Formatting code..."
	$(VENV_CMD) ruff format $(SRC_DIR) $(TEST_DIR)

.PHONY: format-check
format-check: ## Check code formatting without making changes
	@echo "Checking format..."
	$(VENV_CMD) ruff format --check $(SRC_DIR) $(TEST_DIR)

##########################
# Command Line Interface #
##########################

# ACR #

.PHONY: acr-snr
acr-snr:
	$(VENV_CMD) hazen acr_snr $(ACR_DATA)

.PHONY: acr-uniformity
acr-uniformity:
	$(VENV_CMD) hazen acr_uniformity $(ACR_DATA)

.PHONY: acr-ghosting
acr-ghosting:
	$(VENV_CMD) hazen acr_ghosting $(ACR_DATA)

.PHONY: acr-slice-position
acr-slice-position:
	$(VENV_CMD) hazen acr_slice_position $(ACR_DATA)

.PHONY: acr-slice-thickness
acr-slice-thickness:
	$(VENV_CMD) hazen acr_slice_thickness $(ACR_DATA)

.PHONY: acr-geometric-accuracy
acr-geometric-accuracy:
	$(VENV_CMD) hazen acr_geometric_accuracy $(ACR_DATA)

.PHONY: acr-spatial-resolution
acr-spatial-resolution:
	$(VENV_CMD) hazen acr_spatial_resolution $(ACR_DATA)

.PHONY: acr-low-contrast-object-detectability
acr-low-contrast-object-detectability:
	$(VENV_CMD) hazen \
	acr_low_contrast_object_detectability  $(ACR_DATA)

.PHONY: acr-object-detectability
acr-object-detectability:
	$(VENV_CMD) hazen \
	acr_object_detectability  $(ACR_DATA)

.PHONY: acr-large-phantom-all
acr-large-phantom-all:
	$(VENV_CMD) hazen --profile \
	acr_all $(ACR_DATA_T1) $(ACR_DATA_T2) $(ACR_DATA_SL)

.PHONY: cli-acr-all
cli-acr-all: acr-large-phantom-all \
	acr-snr \
	acr-uniformity \
	acr-ghosting \
	acr-slice-position \
	acr-slice-thickness \
	acr-geometric-accuracy \
	acr-spatial-resolution \
	acr-low-contrast-object-detectability \
	acr-object-detectability

.PHONY: cli-acr
cli-acr: cli-acr-all	## Run the ACR CLI tests

# Magnet #
.PHONY: magnet-snr
magnet-snr:
	$(VENV_CMD) hazen snr \
	--measured_slice_width 5.0012 tests/data/snr/Siemens

.PHONY: magnet-uniformity
magnet-uniformity:
	$(VENV_CMD) hazen uniformity tests/data/uniformity

.PHONY: magnet-ghosting
magnet-ghosting:
	$(VENV_CMD) hazen ghosting tests/data/ghosting/GHOSTING

.PHONY: magnet-slice-position
magnet-slice-position:
	$(VENV_CMD) hazen slice_position \
	tests/data/slicepos/SLICEPOSITION

.PHONY: magnet-slice-width
magnet-slice-thickness:
	$(VENV_CMD) hazen slice_width \
	tests/data/slicewidth/512_matrix

.PHONY: magnet-geometric-accuracy
magnet-geometric-accuracy:
	$(VENV_CMD) hazen geometric_accuracy

.PHONY: magnet-spatial-resolution
magnet-spatial-resolution:
	$(VENV_CMD) hazen spatial_resolution \
	tests/data/resolution/RESOLUTION/

.PHONY: magnet-snr-map
magnet-snr-map:
	$(VENV_CMD) hazen snr_map tests/data/snr/Siemens

.PHONY: cli-magnet-all
cli-magnet-all: magnet-snr \
	magnet-uniformity \
	magnet-ghosting \
	magnet-slice-position \
	magnet-slice-width \
	magnet-snr-map \
	magnet-spatial-resolution

.PHONY: cli-magnet
cli-magnet: cli-magnet-all	## Run the MagNET CLI tests

# Calibre #

.PHONY: relaxometry-T1
relaxometry-T1:
	$(VENV_CMD) hazen relaxometry \
	tests/data/relaxometry/T1/site1_20200218/plate5 \
	--calc T1 --plate_number=5

.PHONY: relaxometry-T2
relaxometry-T2:
	$(VENV_CMD) hazen relaxometry \
	tests/data/relaxometry/T2/site3_ge/plate4 \
	--calc T2 --plate_number=4

.PHONY: caliber-relaxometry
caliber-relaxometry: relaxometry-T1 relaxometry-T2

.PHONY: cli-caliber
cli-caliber: caliber-relaxometry	## Run the Caliber CLI tests

.PHONY: cli
cli: cli-acr cli-magnet cli-caliber	## Run all CLI tests

###################
# Combined Checks #
###################

.PHONY: check-notypes
check-notypes: format-check test cli	## Run checks without types

.PHONY: check
check: type-check check-notypes

.PHONY: ci
ci: check ## Run full CI pipeline

#################
# Documentation #
#################

.PHONY: docs
docs: ## Build documentation
	@echo "Building documentation..."
	$(VENV_CMD) sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html

.PHONY: docs-serve
docs-serve: docs ## Build and serve documentation
	@echo "Serving documentation on http://localhost:8000..."
	$(VENV_CMD) python -m http.server -d $(DOCS_DIR)/_build/html

##################
# Build & Deploy #
##################

.PHONY: build
build: ## Build the package
	@echo "Building package..."
	uv build

#########
# Clean #
#########

.PHONY: clean
clean: ## Clean build artifacts and caches
	@echo "Cleaning..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/
	rm -rf build/ dist/ *.egg-info
	rm -rf $(DOCS_DIR)/_build
	@echo "Clean complete"

###############
# Development #
###############

.PHONY: dev-install
dev-install: ## Install with development dependencies
	@echo "Installing development dependencies..."
	uv sync --all-extras

.PHONY: pre-commit
pre-commit: ## Install pre-commit hook
	@echo "Setting up pre-commit hook..."
	echo '#!/bin/sh\nmake ci' > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed"
