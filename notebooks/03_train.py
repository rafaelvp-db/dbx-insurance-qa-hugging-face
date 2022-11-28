# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Distilbert Example

# COMMAND ----------

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

# COMMAND ----------

# DBTITLE 1,Customizing our Model
# MAGIC %md
# MAGIC 
# MAGIC * Our model must have 12 different labels; we need to customize `DistilBERT` to support that
# MAGIC * But first, we need to have a Torch `Dataset` and a `DataLoader`. Let's code that up

# COMMAND ----------

import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.storagelevel import StorageLevel
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

class InsuranceDataset(Dataset):
    def __init__(
      self,
      database_name = "insuranceqa",
      split = "train",
      input_col = "question_en",
      label_col = "topic_en",
      storage_level = StorageLevel.MEMORY_ONLY,
      tokenizer = "distilbert-base-uncased",
      max_length = 50
    ):
      self.input_col = input_col
      self.label_col = label_col
      self.df = spark.sql(
        f"select * from {database_name}.{split}"
      ).persist(storage_level)
      self.length = self.df.count()
      self.class_mappings = self._get_class_mappings()
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
      self.max_length = max_length

    def _get_class_mappings(self):
      labels = self.df \
        .select(f"topic_en") \
        .distinct() \
        .toPandas() \
        .iloc[:, 0]

      indexes = LabelEncoder().fit_transform(labels)
      class_mappings = dict(zip(labels, indexes))
      return class_mappings

    def __len__(self):
      return self.length

    def __getitem__(self, idx):
      query = f"id = {idx}"
      df = self.df.filter(query).toPandas()
      question = df.loc[0, self.input_col]
      inputs = self.tokenizer(
        question,
        add_special_tokens=True,
        max_length=self.max_length,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True
      )

      ids = inputs['input_ids']
      mask = inputs['attention_mask']
      label = self.class_mappings[df.loc[0, self.label_col]]

      return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'targets': torch.tensor(label, dtype=torch.long)
      }

# COMMAND ----------

from torch.utils.data import DataLoader

training_data = InsuranceDataset(split = "train")
test_data = InsuranceDataset(split = "test")

training_data[0]

# COMMAND ----------

# DBTITLE 1,Instantiating Data Loaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

# COMMAND ----------

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

class LitModel(pl.LightningModule):
    def __init__(self, model = "distilbert-base-uncased"):
      super().__init__()
      self.l1 = DistilBertForSequenceClassification.from_pretrained(model)
      self.pre_classifier = torch.nn.Linear(768, 768)
      self.dropout = torch.nn.Dropout(0.3)
      self.classifier = torch.nn.Linear(768, 12)

    def forward(self, sample):
      print(sample)
      output_1 = self.l1(
        input_ids = sample["ids"],
        attention_mask = sample["mask"]
      )
      hidden_state = output_1[0]
      pooler = hidden_state[:, 0]
      pooler = self.pre_classifier(pooler)
      pooler = torch.nn.ReLU()(pooler)
      pooler = self.dropout(pooler)
      output = self.classifier(pooler)
      return output

    def training_step(self, batch, batch_idx):
      x, y = batch
      y_hat = self(x)
      loss = F.cross_entropy(y_hat, y)
      return loss

    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=0.02)

# COMMAND ----------

lit_model = LitModel()
trainer = pl.Trainer(accelerator = "gpu")

# COMMAND ----------

lit_model(next(iter(training_data)))

# COMMAND ----------

trainer.fit(lit_model, train_dataloader, test_dataloader)

# COMMAND ----------


