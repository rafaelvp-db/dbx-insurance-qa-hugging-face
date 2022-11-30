# Insurance Q&A Intent Classification with Databricks & Hugging Face

**TLDR;** this repo contains code that showcases the process of:
* Ingesting data related to Insurance questions and answers ([Insurance QA Dataset](https://github.com/shuzi/insuranceQA)) into Delta Lake
* Basic cleaning and preprocessing
* Creating custom [PyTorch Lightning](https://www.pytorchlightning.ai/) `DataModule` and `LightningModule` to wrap, respectively, our dataset and our backbone model (`distilbert_en_uncased`)
* Training with multiple GPUs while logging desired metrics into MLflow and registering model assets into Databricks Model Registry
* Running inference both with single and multiple nodes
