clean:
	rm -rf spark-warehouse && \
	rm -rf mlruns && \
	rm -rf 0

env:
	pip install -e '.[local, test]'

unit: clean
	pytest tests/unit

format:
	black insuranceqa

lint:
	flake8 insuranceqa

integration: clean
	pytest tests/integration

ingest:
	dbx execute --cluster-id=1021-161236-mcec790m insuranceqa-multitask --task ingest
	