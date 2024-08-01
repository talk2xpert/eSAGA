from pyspark.sql import SparkSession
import  pandas as pd
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = (SparkSession.builder.appName("Datacamp Pyspark Tutorial").config("spark.memory.offHeap.enabled","true")
         .config("spark.memory.offHeap.size","10g").getOrCreate())
#pandas_df = pd.read_excel("C:\dataset\\datacamp_ecommerce.xlsx")
#pandas_df.to_csv('C:\dataset\\datacamp_ecommerce_csv.csv', index=False)
# Read the CSV file
spark_df = spark.read.csv('C:\dataset\\datacamp_ecommerce_csv.csv', header=True, inferSchema=True)

# Show the DataFrame
#print(spark_df.show())

print(spark_df.groupBy('Country').agg(countDistinct('CustomerID').alias('country_count')).show())
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
#To find when the latest purchase was made on the platform
spark_df = spark_df.withColumn('date',to_timestamp("InvoiceDate", 'yy/MM/dd HH:mm'))
spark_df.select(max("date")).show()
#df.select(min("date")).show()

spark_df = spark_df.withColumn("from_date", lit("12/1/10 08:26"))
spark_df = spark_df.withColumn('from_date',to_timestamp("from_date", 'yy/MM/dd HH:mm'))

df2=spark_df.withColumn('from_date',to_timestamp(col('from_date'))).withColumn('recency',col("date").cast("long") - col('from_date').cast("long"))

df2 = df2.join(df2.groupBy('CustomerID').agg(max('recency').alias('recency')),on='recency',how='leftsemi')
df2.show(5,0)
df2.printSchema()
