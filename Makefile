clean:
	rm -rf spark-warehouse && \
	rm -rf mlruns && \
	rm -rf 0

env:
	pip install -e '.[local, test]'

unit: clean
	pytest tests/unit

integration: clean
	pytest tests/integration
	