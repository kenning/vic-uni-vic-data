import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    NGram,
    Normalizer,
)
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier

SEED = 100
USE_L2_NORM = True

username = sys.argv[1]
test_run = False
if username == "test":
    test_run = True
    print("Test run")
else:
    print(f"Running as user {username}")

small_sample_and_dt = False
sample_fraction = 0.0001
if len(sys.argv) > 2:
    if sys.argv[2] == "--sample":
        small_sample_and_dt = True

        print(f"Only running on sample of {sample_fraction*100:02}% of data")


def run():
    start_time = time.time()

    spark = SparkSession.builder.appName("AuthorPrediction").getOrCreate()
    sc = spark.sparkContext

    ############################################################
    # Data ingestion
    ############################################################

    data = (
        spark.read.option("header", "true")
        .option("encoding", "latin1")
        .option("inferSchema", "true")
        .csv(
            "./Gungor_2018_VictorianAuthorAttribution_data.csv",
        )
    )

    data = data.withColumn("author", col("author").cast(IntegerType()))

    # For local testing: take a small subset of data
    if small_sample_and_dt:
        sample = data.sample(withReplacement=False, fraction=sample_fraction, seed=SEED)
    else:
        sample = data

    (training_data, test_data) = sample.randomSplit([0.8, 0.2], seed=SEED)

    ############################################################
    # Pipeline component initialization
    ############################################################

    # Weird issue with the data where "â" is the most common word,
    # seems to be a unicode stop character related to text encoding
    stop_words = StopWordsRemover.loadDefaultStopWords("english")
    custom_stop_words = stop_words + ["â"]

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(
        inputCol="words", outputCol="filtered", stopWords=custom_stop_words
    )

    bigram = NGram(n=2, inputCol="filtered", outputCol="bigrams")
    trigram = NGram(n=3, inputCol="filtered", outputCol="trigrams")

    unigram_vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
    bigram_vectorizer = CountVectorizer(inputCol="bigrams", outputCol="features")
    trigram_vectorizer = CountVectorizer(inputCol="trigrams", outputCol="features")

    ############################################################
    # Run DT and LR models
    ############################################################

    kwargz = {
        "spark": spark,
        "tokenizer": tokenizer,
        "remover": remover,
        "unigram_vectorizer": unigram_vectorizer,
        "bigram": bigram,
        "bigram_vectorizer": bigram_vectorizer,
        "trigram": trigram,
        "trigram_vectorizer": trigram_vectorizer,
        "training_data": training_data,
        "test_data": test_data,
        "is_dt": False,
    }

    lr_logs = build_pipeline_and_evaluate(**kwargz)

    ############################################################
    # Print results, optionally to HDFS
    ############################################################

    time_log = [f"Full process took {time.time() - start_time} seconds"]
    all_logs = lr_logs + time_log
    print("\n".join(all_logs))

    print(time_log)

    if not test_run:
        rdd = sc.parallelize(all_logs, numSlices=1)
        log_output_path = (
            f"hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output/logs"
        )
        rdd.saveAsTextFile(log_output_path)


############################################################
# Builds either LR or DT pipeline and evaluates data.
# Also generates decision tree, as well as ngram dataframe.
############################################################
def build_pipeline_and_evaluate(
    spark,
    tokenizer,
    remover,
    unigram_vectorizer,
    bigram,
    bigram_vectorizer,
    trigram,
    trigram_vectorizer,
    training_data,
    test_data,
    is_dt,
):
    result_strings = []
    classifier_name = "Logistic Regression"

    stages = [tokenizer, remover, unigram_vectorizer]
    features_col_name = "features"
    if USE_L2_NORM:
        features_col_name = "norm_features"
        # p=2.0 == L2 norm
        normalizer = Normalizer(inputCol="features", outputCol=features_col_name, p=2.0)
        stages.append(normalizer)

    lr = LogisticRegression(featuresCol=features_col_name, labelCol="author")
    dt = DecisionTreeClassifier(featuresCol=features_col_name, labelCol="author")
    classifier = dt if is_dt else lr
    stages.append(classifier)
    curr_pipeline = Pipeline(stages=stages)
    curr_pipeline_name = "Unigram"

    curr_model = curr_pipeline.fit(training_data)
    for testing_set in [test_data, training_data]:
        test_set_name = "Test" if testing_set == test_data else "Train"
        curr_predictions = curr_model.transform(test_data)

        curr_evaluator = MulticlassClassificationEvaluator(
            labelCol="author", predictionCol="prediction", metricName="accuracy"
        )
        curr_accuracy = curr_evaluator.evaluate(curr_predictions)
        result_string = f"[{classifier_name} {test_set_name}] Accuracy for {curr_pipeline_name}: {curr_accuracy:.5f}"
        print(result_string)
        result_strings.append(result_string)

    return result_strings


############################################################
# Helper function to save csv with predictable filename
############################################################
def save_df_to_hdfs(spark, df, hdfs_path, filename):
    # https://stackoverflow.com/questions/69635571/how-to-save-a-pyspark-dataframe-as-a-csv-with-custom-file-name

    sc = spark.sparkContext
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration
    df.coalesce(1).write.option("header", True).option("delimiter", ",").option(
        "compression", "none"
    ).mode("overwrite").csv(hdfs_path)
    fs = FileSystem.get(Configuration())
    file = fs.globStatus(Path("%s/part*" % hdfs_path))[0].getPath().getName()
    full_path = "%s/%s" % (hdfs_path, filename)
    result = fs.rename(Path("%s/%s" % (hdfs_path, file)), Path(full_path))
    return result


if __name__ == "__main__":
    print("Running Victorian Author analysis...")
    run()
    print("--\n--\nDone!\n--\n--")
