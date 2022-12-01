import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope = "session")
def spark():
    """PySpark session fixture."""

    spark = SparkSession.builder.getOrCreate()
    return spark