.PHONY: test lint clean

test:
	python -m pytest workspace/tests/ -v

lint:
	python -m flake8 workspace/src/

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
