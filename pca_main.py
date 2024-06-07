import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

spark = SparkSession.builder.appName("VictorianAuthorAttribution").getOrCreate()

username = sys.argv[1]


def evaluate_logistic_regression(file_path, pca_components, ngrams):
    start = time.time()
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    feature_columns = [f"PC{i+1}" for i in range(pca_components)]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    train, test = df.randomSplit([0.8, 0.2], seed=100)

    lr = LogisticRegression(featuresCol="features", labelCol="author")

    lr_model = lr.fit(train)

    predictions = lr_model.transform(test)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="author", predictionCol="prediction", metricName="accuracy"
    )
    train_accuracy = evaluator.evaluate(predictions)

    predictions = lr_model.transform(train)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="author", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)

    return [
        f"--"
        f"Analyzing {ngrams} PCA dataset with {pca_components} components"
        f"Test Accuracy {train_accuracy:.4f}",
        f"Train Accuracy {accuracy:.4f}",
        f"Total time elapsed: {time.time() - start} seconds",
        "--",
    ]


file_paths = [
    ("victorian_author_attribution_pca_2.csv", 2),
    ("victorian_author_attribution_pca_20.csv", 20),
    ("victorian_author_attribution_pca_200.csv", 200),
    ("2gram_victorian_author_attribution_pca_2.csv", 2),
    ("3gram_victorian_author_attribution_pca_2.csv", 2),
    ("2gram_victorian_author_attribution_pca_10.csv", 10),
    ("1gram_victorian_author_attribution_pca_10.csv", 10),
]

all_logs = []
for file_path, pca_components in file_paths:
    all_logs = all_logs + evaluate_logistic_regression(file_path, pca_components)

sc = spark.sparkContext
rdd = sc.parallelize(all_logs, numSlices=1)
log_output_path = f"hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output/logs"
rdd.saveAsTextFile(log_output_path)

spark.stop()
