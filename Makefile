local:
	pip install -e '.[local]'

integration:
	pytest tests/integration
	