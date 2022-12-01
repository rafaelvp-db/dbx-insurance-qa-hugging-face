from insuranceqa.tasks.ingest import IngestionTask
from fixtures.spark import spark
from pyspark.sql import SparkSession
from pathlib import Path
import logging

def test_ingestion_job(spark: SparkSession, tmp_path: Path):
    logging.info("Testing the ingestion job")
    common_config = {"database": "insuranceqa"}
    test_ingest_config = {"output": common_config}
    ingestion_job = IngestionTask(spark, test_ingest_config)
    ingestion_job.launch()
    table_name = f"{test_ingest_config['output']['database']}.train"
    _count = spark.table(table_name).count()
    assert _count > 0
    logging.info("Testing the ingestion job - done")


