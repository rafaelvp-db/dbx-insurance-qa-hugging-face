import pytest
from pyspark.sql import SparkSession


@pytest.fixture
def spark():
    """PySpark session fixture."""
    spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()