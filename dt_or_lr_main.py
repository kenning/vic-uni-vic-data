import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, NGram
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier


RUN_LR = True
RUN_DT = False
DT_DEPTH = 30

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
        sample = data.sample(withReplacement=False, fraction=sample_fraction, seed=100)
    else:
        sample = data

    (training_data, test_data) = sample.randomSplit([0.8, 0.2], seed=100)

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

    unigram_vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")

    ############################################################
    # Run DT and LR models
    ############################################################

    kwargz = {
        "spark": spark,
        "tokenizer": tokenizer,
        "remover": remover,
        "unigram_vectorizer": unigram_vectorizer,
        "training_data": training_data,
        "test_data": test_data,
        "is_dt": False,
    }

    if RUN_LR:
        lr_logs, _, _ = build_pipeline_and_evaluate(**kwargz)
    else:
        lr_logs = ["test lr log"]

    if RUN_DT:
        kwargz["is_dt"] = True
        dt_logs, dt_strings, feature_dfs = build_pipeline_and_evaluate(**kwargz)
    else:
        dt_logs = []
        dt_strings = []
        feature_dfs = None

    ############################################################
    # Print results, optionally to HDFS
    ############################################################

    time_log = [f"Full process took {time.time() - start_time} seconds"]
    all_logs = dt_strings + lr_logs + dt_logs + time_log
    print("\n".join(all_logs))
    if feature_dfs is not None:
        for df in feature_dfs:
            df.show()

    print(time_log)

    if not test_run:
        rdd = sc.parallelize(all_logs, numSlices=1)
        log_output_path = (
            f"hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output/logs"
        )
        rdd.saveAsTextFile(log_output_path)

        if feature_dfs is not None:
            for i, df in enumerate(feature_dfs):
                output_path = f"hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output/df-{i}"
                save_df_to_hdfs(spark, df, output_path, f"feat_df_{i}.csv")


############################################################
# Builds either LR or DT pipeline and evaluates data.
# Also generates decision tree, as well as ngram dataframe.
############################################################
def build_pipeline_and_evaluate(
    spark,
    tokenizer,
    remover,
    unigram_vectorizer,
    training_data,
    test_data,
    is_dt,
):
    features_col = "features"

    lr = LogisticRegression(featuresCol=features_col, labelCol="author")
    dt = DecisionTreeClassifier(
        featuresCol=features_col, labelCol="author", maxDepth=DT_DEPTH
    )
    classifier = dt if is_dt else lr
    classifier_name = "Decision Tree" if is_dt else "Logistic Regression"
    result_strings = []
    decision_tree_strings = []
    feature_dfs = []

    stages = [tokenizer, remover, unigram_vectorizer]
    curr_pipeline_name = "Unigram"

    # Add classifier stage
    stages.append(classifier)

    curr_pipeline = Pipeline(stages=stages)

    curr_model = curr_pipeline.fit(training_data)
    for testing_set in [test_data, training_data]:
        test_set_name = "Test" if testing_set == test_data else "Train"
        curr_predictions = curr_model.transform(testing_set)

        curr_evaluator = MulticlassClassificationEvaluator(
            labelCol="author", predictionCol="prediction", metricName="accuracy"
        )
        curr_accuracy = curr_evaluator.evaluate(curr_predictions)
        result_string = f"[{classifier_name} {test_set_name}] Accuracy for {curr_pipeline_name}: {curr_accuracy:.5f}"
        print(result_string)
        result_strings.append(result_string)

        if is_dt:
            dt_str, feature_df = create_dt_and_feature_df(
                spark=spark, dt_model=curr_model
            )
            decision_tree_strings.append(dt_str)
            feature_dfs.append(feature_df)

    return result_strings, decision_tree_strings, feature_dfs


############################################################
# Generate decision tree and ngram dataframe.
############################################################
def create_dt_and_feature_df(spark, dt_model):
    dt_classifier_model = dt_model.stages[-1]
    dt_vectorizer = dt_model.stages[-2]
    dt_vocabulary = dt_vectorizer.vocabulary
    tree_string = dt_classifier_model.toDebugString

    import re
    from collections import Counter
    from pyspark.sql import SparkSession

    def replace_and_collect_feature_indices(tree_string, vocabulary):
        feature_indices = []

        def replace_match(match):
            feature_index = int(match.group(1))
            feature_name = vocabulary[feature_index]
            feature_indices.append(feature_index)
            return f"feature ({feature_index}) [{feature_name}]"

        tree_string_with_names = re.sub(r"feature (\d+)", replace_match, tree_string)
        return tree_string_with_names, feature_indices

    tree_string_with_names, feature_indices = replace_and_collect_feature_indices(
        tree_string, dt_vocabulary
    )

    feature_counts = Counter(feature_indices)

    feature_usage_data = [
        (dt_vocabulary[idx], count) for idx, count in feature_counts.items()
    ]

    feature_usage_df = spark.createDataFrame(
        feature_usage_data, ["feature_name", "count"]
    )

    feature_usage_df.orderBy(col("count").desc()).show()

    return tree_string_with_names, feature_usage_df


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
