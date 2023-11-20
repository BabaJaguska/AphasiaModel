# Define variables
LINTER = flake8
TEST_DIR = ./tests
PROFILE_SCRIPT = ./tests/performance.py

# Default target
all: install format lint test profile

# Install dependencies
install:
	pip install -r requirements.txt

# Format code
format:
	black .

# Lint code
lint:
	$(LINTER) --max-line-length=88 --ignore=E203,E501 .

# Run tests
test:
	coverage run -m unittest discover -s $(TEST_DIR)

# Check the test coverage
cover:
	coverage report -m

# Run profiling
profile:
	PYTHONPATH=./ python $(PROFILE_SCRIPT)

# Help command to display available commands (optional)
help:
	@echo "Available commands:"
	@echo "  all     - Install dependencies, format, lint, test and profile"
	@echo "  install - Install Python dependencies"
	@echo "  format  - Format Python code"
	@echo "  lint    - Lint Python code"
	@echo "  test    - Run tests"
	@echo "  profile - Run profiling script"
	@echo "  help    - Display this help message"

.PHONY: all install format lint test profile cover help
