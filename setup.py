"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from insuranceqa import __version__

PACKAGE_REQUIREMENTS = [
    "pyyaml"
]

# packages for local development and unit testing
# please note that these packages are already available in DBR, there is no need to install them on DBR.
LOCAL_REQUIREMENTS = [
    "pyspark==3.2.1",
    "delta-spark==1.1.0",
    "transformers",
    "torch",
    "pytorch_lightning",
    "scikit-learn",
    "pandas",
    "mlflow"
]

TEST_REQUIREMENTS = [
    # development & testing tools
    "pytest",
    "coverage[toml]",
    "pytest-cov",
    "dbx>=0.7,<0.8",
    "flake8",
    "black"
]

setup(
    name="insuranceqa",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["setuptools","wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"local": LOCAL_REQUIREMENTS, "test": TEST_REQUIREMENTS},
    entry_points = {
        "console_scripts": [
            "ingest = insuranceqa.tasks.ingest:entrypoint",
            "clean = insuranceqa.tasks.clean:entrypoint",
            "train = insuranceqa.tasks.train:entrypoint"
    ]},
    version=__version__,
    description="",
    author="https://www.github.com/rafaelvp-db",
)
