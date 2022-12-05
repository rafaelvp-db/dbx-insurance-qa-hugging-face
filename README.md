# Insurance Q&A Intent Classification with Databricks & Hugging Face

<img src="https://github.com/rafaelvp-db/dbx-insurance-qa-hugging-face/blob/master/img/header.png?raw=true" />

<hr />

**TLDR;** this repo contains code that showcases the process of:
* Ingesting data related to Insurance questions and answers ([Insurance QA Dataset](https://github.com/shuzi/insuranceQA)) into Delta Lake
* Basic cleaning and preprocessing
* Creating custom [PyTorch Lightning](https://www.pytorchlightning.ai/) `DataModule` and `LightningModule` to wrap, respectively, our dataset and our backbone model (`distilbert_en_uncased`)
* Training with multiple GPUs while logging desired metrics into MLflow and registering model assets into Databricks Model Registry
* Running inference both with single and multiple nodes


## Additional Reference

1. Minwei Feng, Bing Xiang, Michael R. Glass, Lidan Wang, Bowen Zhou. [Applying Deep Learning to Answer Selection: A Study and An Open Task](https://arxiv.org/abs/1508.01585)
2. [Fine-tune Transformers Models with PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html)
3. [PyTorch Lightning MLflow Logger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.mlflow.html)
