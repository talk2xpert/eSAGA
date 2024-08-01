from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("MongoSparkExample") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.11:10.0.5") \
    .config("spark.mongodb.input.uri", "=mongodb://localhost:27017//workmateCapsule") \
    .config("spark.mongodb.output.uri", "=mongodb://localhost:27017//workmateCapsule") \
    .getOrCreate()

# Read data from MongoDB
df = spark.read.format("ms_account_configuration").load()
