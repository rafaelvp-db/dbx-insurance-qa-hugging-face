from insuranceqa.tasks.ingest import IngestionTask
from fixtures.spark import spark
from pyspark.sql import SparkSession
from pathlib import Path
import logging

def test_cleaning_job(spark: SparkSession, tmp_path: Path):
    logging.info("Testing the cleaning job")
    test_clean_config = {
        "filter": "length < 50",
        "input": {"database": "insuranceqa"},
        "output": {"suffix": "silver"}
    }

    ingestion_job = IngestionTask(spark, test_clean_config)
    ingestion_job.launch()
    logging.info("Testing the cleaning job - done")
    pass


