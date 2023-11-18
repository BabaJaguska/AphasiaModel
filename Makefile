# Define variables
LINTER = flake8
TEST_DIR = ./tests  
PROFILE_SCRIPT = ./test/profile.py

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
	$(LINTER) .

# Run tests
test:
	python -m unittest discover -s $(TEST_DIR)

# Run profiling
profile:
	python $(PROFILE_SCRIPT)

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

.PHONY: all install format lint test profile help