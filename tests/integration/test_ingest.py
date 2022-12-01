from insuranceqa.data.ingest import InsuranceQA
import logging
from pyspark.sql import SparkSession

def test_download_ingest():
    
    spark = SparkSession.builder.getOrCreate()
    
    try:
        logging.info("Downloading data")
        ds = InsuranceQA(spark)
        logging.info("Ingesting into Delta")
        ds.ingest()
        pass
    except Exception as excp:
        logging.error(f"Error: {str(excp)}")
        assert False
    finally:
        spark.stop()
