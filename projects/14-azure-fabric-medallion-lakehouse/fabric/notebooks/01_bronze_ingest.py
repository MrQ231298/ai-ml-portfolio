# Fabric notebook (PySpark) - BRONZE ingest
# STATUS: NOT EXECUTED. Written as the Fabric/Spark equivalent of src/bronze.py;
# there was no Fabric workspace or Java runtime on the build machine.
# Attach to a Lakehouse; landing files are in Files/landing/online_retail_ii/.
from pyspark.sql import functions as F
from delta.tables import DeltaTable

LANDING = "Files/landing/online_retail_ii"
BRONZE = "bronze_online_retail_ii_raw"
CONTROL = "ingested_files"
batch_id = spark.conf.get("spark.app.id")  # or a pipeline parameter

files = [f for f in mssparkutils.fs.ls(LANDING) if f.name.endswith(".csv")]  # noqa: F821
known = {r.file_name: r.sha256 for r in spark.table(CONTROL).collect()} if spark.catalog.tableExists(CONTROL) else {}

for f in files:
    df = (spark.read.option("header", True).option("inferSchema", False).csv(f.path)   # everything stays string
          .toDF("invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"))
    sha = df.select(F.sha2(F.concat_ws("|", *df.columns), 256).alias("h")).agg(F.sha2(F.concat_ws("", F.collect_list("h")), 256)).first()[0]
    if known.get(f.name) == sha:
        continue
    month = f.name.removesuffix(".csv")
    out = (df.withColumn("_source_file", F.lit(f.name)).withColumn("_source_month", F.lit(month))
             .withColumn("_source_sha256", F.lit(sha)).withColumn("_batch_id", F.lit(batch_id)))
    (out.write.format("delta").mode("overwrite").option("replaceWhere", f"_source_month = '{month}'")
        .partitionBy("_source_month").saveAsTable(BRONZE))
    spark.createDataFrame([(f.name, sha, batch_id, out.count())], "file_name string, sha256 string, batch_id string, rows long") \
         .createOrReplaceTempView("new_ctl")
    if spark.catalog.tableExists(CONTROL):
        DeltaTable.forName(spark, CONTROL).alias("t").merge(spark.table("new_ctl").alias("s"), "t.file_name = s.file_name") \
            .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    else:
        spark.table("new_ctl").write.format("delta").saveAsTable(CONTROL)
