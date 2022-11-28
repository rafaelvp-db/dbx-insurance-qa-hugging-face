# Databricks notebook source
!rm -rf /dbfs/tmp/word2vec-get-started
%cd /tmp
!git clone https://github.com/hailiang-wang/word2vec-get-started.git
!mv /tmp/word2vec-get-started/corpus /dbfs/tmp/word2vec-get-started

# COMMAND ----------

dbfs_path = "/dbfs/tmp/word2vec-get-started/insuranceqa/questions"
!ls -all {dbfs_path}

# COMMAND ----------

from pyspark.sql.functions import lower, regexp_replace, col

def ingest_data(
  path,
  output_table
):

  spark.sql("create database if not exists insuranceqa")
  spark.sql(f"drop table if exists {output_table}")

  df = spark.read.csv(
    path,
    sep = "\t",
    header = True
  )

  df = df.toDF(
    'id',
    'topic_en',
    'topic_jp',
    'question_en',
    'question_jp'
  )\
  .select("id", "topic_en", "question_en")

  return df

def clean(df):

  df = df.withColumn("question_en", regexp_replace(lower(col("question_en")), "  ", " "))
  return df

def pipeline(path, output_table):
  df = ingest_data(path, output_table)
  df = clean(df)
  df.write.saveAsTable(output_table)

pipeline(f"{dbfs_path.replace('/dbfs', '')}/train.questions.txt", "insuranceqa.train")
pipeline(f"{dbfs_path.replace('/dbfs', '')}/test.questions.txt", "insuranceqa.test")
pipeline(f"{dbfs_path.replace('/dbfs', '')}/valid.questions.txt", "insuranceqa.valid")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from insuranceqa.train limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from insuranceqa.test limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from insuranceqa.valid limit 10
