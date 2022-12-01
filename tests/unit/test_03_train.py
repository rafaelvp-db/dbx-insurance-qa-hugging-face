from insuranceqa.tasks.train import TrainTask
from pathlib import Path
import logging
from fixtures.spark import spark
from pyspark.sql import SparkSession
import os


def test_training_job(spark: SparkSession, tmp_path: Path):
    logging.info("Testing the cleaning job")
    test_train_config = {
        "accelerator": "cpu",
        "devices": 1,
        "max_steps": 1,
        "input": {
            "suffix": "silver",
            "train": "train",
            "test": "test",
            "valid": "valid"
        },
        "num_workers": 10,
        "tracking_uri": "./mlruns",
        "batch_size": 1
    }

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    training_job = TrainTask(
        spark = spark,
        init_conf = test_train_config
    )
    training_job.launch()
    logging.info("Testing the training job - done")
    pass


