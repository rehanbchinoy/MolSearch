# Molecular Search Pipeline Makefile

.PHONY: help install install-dev test test-cov lint format clean run-example setup-dirs

# Default target
help:
	@echo "Molecular Search Pipeline - Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup-dirs   - Create necessary directories"
	@echo ""
	@echo "Development:"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean up generated files"
	@echo ""
	@echo "Usage:"
	@echo "  run-example  - Run example with fentanyl"
	@echo "  cli-help     - Show CLI help"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

setup-dirs:
	mkdir -p data models output logs tests

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=molsearch_pipeline --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 molsearch_pipeline.py cli.py tests/
	mypy molsearch_pipeline.py cli.py

format:
	black molsearch_pipeline.py cli.py tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf output/*
	rm -rf logs/*

# Example usage
run-example:
	python cli.py --smiles "CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3" --verbose

cli-help:
	python cli.py --help

# Development workflow
dev-setup: setup-dirs install-dev
	@echo "Development environment setup complete!"

# Quick test
quick-test:
	python -c "from molsearch_pipeline import Config; print('Import successful!')"

# Check dependencies
check-deps:
	python -c "import rdkit, datamol, molfeat, pinecone; print('All dependencies available!')"

# Create virtual environment
venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"

# Install in development mode
install-dev-mode:
	pip install -e .

# Run with different configurations
run-rdkit:
	python cli.py --smiles "CCO" --featurizer rdkit_2d --variations 10

run-chemgpt:
	python cli.py --smiles "CCO" --featurizer chemgpt --variations 10

# Performance test
perf-test:
	python -c "
import time
from molsearch_pipeline import Config, MolecularFeaturizer
from rdkit import Chem

config = Config()
featurizer = MolecularFeaturizer(config)
mols = [Chem.MolFromSmiles('CCO') for _ in range(100)]

start = time.time()
features = featurizer.featurize_molecules(mols)
end = time.time()

print(f'Featurized 100 molecules in {end-start:.2f} seconds')
print(f'Features shape: {features.shape}')
"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "See README.md for comprehensive documentation"

# Docker (if needed)
docker-build:
	docker build -t molsearch-pipeline .

docker-run:
	docker run -it molsearch-pipeline python cli.py --smiles "CCO"

# Environment setup
env-setup:
	@echo "Setting up environment variables..."
	@if [ ! -f .env ]; then \
		echo "PINECONE_API_KEY=your_key_here" > .env; \
		echo "Created .env file. Please update with your API keys."; \
	else \
		echo ".env file already exists."; \
	fi

# Full setup
full-setup: venv env-setup setup-dirs install-dev
	@echo "Full setup complete!"
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Update .env with your API keys"
	@echo "3. Run: make run-example" 