rm:
	rm -rf spark-warehouse && \
	rm -rf mlruns && \
	rm -rf 0

env:
	pip install -e '.[local, test]'

<<<<<<< HEAD
unit: clean
=======
unit: rm
>>>>>>> 934fadb (cleanup)
	pytest tests/unit

format:
	black insuranceqa

lint:
	flake8 insuranceqa

<<<<<<< HEAD
integration: clean
=======
integration: rm
>>>>>>> 934fadb (cleanup)
	pytest tests/integration

ingest:
	dbx execute --cluster-id=1021-161236-mcec790m insuranceqa-multitask --task ingest

clean:
	dbx execute --cluster-id=1021-161236-mcec790m insuranceqa-multitask --task clean

train:
	dbx execute --cluster-id=1011-090100-bait793 insuranceqa-multitask --task train