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
    VectorAssembler,
)
from pyspark.ml.classification import DecisionTreeClassifier


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

    bigram = NGram(n=2, inputCol="filtered", outputCol="bigrams")

    ############################################################
    # Run DT model
    ############################################################

    dt_logs, dt_strings, feature_dfs = build_and_eval_unigram_and_bigram_pipeline(
        spark=spark,
        tokenizer=tokenizer,
        remover=remover,
        bigram=bigram,
        training_data=training_data,
        test_data=test_data,
    )

    ############################################################
    # Print results, optionally to HDFS
    ############################################################

    time_log = [f"Full process took {time.time() - start_time} seconds"]
    all_logs = dt_logs + dt_strings + time_log
    print("\n".join(all_logs))
    for df in feature_dfs:
        df.show()

    print(time_log)

    if not test_run:
        rdd = sc.parallelize(all_logs, numSlices=1)
        output_path = f"hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output"
        rdd.saveAsTextFile(output_path)

        for i, df in enumerate(feature_dfs):
            save_df_to_hdfs(spark, df, output_path, f"feat_df_{i}.csv")


def build_and_eval_unigram_and_bigram_pipeline(
    spark,
    tokenizer,
    remover,
    bigram,
    assembler,
    training_data,
    test_data,
):
    # Need different vectorizers for combination later
    both_unigram_vectorizer = CountVectorizer(
        inputCol="filtered", outputCol="unigram_features"
    )
    both_bigram_vectorizer = CountVectorizer(
        inputCol="bigrams", outputCol="bigram_features"
    )
    assembler = VectorAssembler(
        inputCols=["unigram_features", "bigram_features"], outputCol="features"
    )

    classifier = DecisionTreeClassifier(featuresCol="features", labelCol="author")

    result_strings = []
    decision_tree_strings = []
    feature_dfs = []

    both_pipeline = Pipeline(
        stages=[
            tokenizer,
            remover,
            bigram,
            both_unigram_vectorizer,
            both_bigram_vectorizer,
            assembler,
            classifier,
        ]
    )

    both_model = both_pipeline.fit(training_data)
    both_predictions = both_model.transform(test_data)

    unigram_vocab = both_model.stages[3].vocabulary
    bigram_vocab = both_model.stages[4].vocabulary
    combined_vocab = unigram_vocab + bigram_vocab

    # Evaluate unigrams+bigrams
    both_evaluator = MulticlassClassificationEvaluator(
        labelCol="author", predictionCol="prediction", metricName="accuracy"
    )
    both_accuracy = both_evaluator.evaluate(both_predictions)
    result_string = f"[Decision Tree] Test Accuracy: {both_accuracy:.5f}"
    print(result_string)
    result_strings.append(result_string)

    # Run extra DT analysis
    dt_str, feature_df = create_both_dt_and_dfs(
        spark=spark, dt_model=both_model, vocab=combined_vocab
    )
    decision_tree_strings.append(dt_str)
    feature_dfs.append(feature_df)

    return result_strings, decision_tree_strings, feature_dfs


def create_both_dt_and_dfs(spark, dt_model, vocab):
    dt_classifier_model = dt_model.stages[-1]
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
        tree_string, vocab
    )

    feature_counts = Counter(feature_indices)

    feature_usage_data = [(vocab[idx], count) for idx, count in feature_counts.items()]

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
    df.coalesce(1).write.option("header", True).option("delimiter", "|").option(
        "compression", "none"
    ).csv(hdfs_path)
    fs = FileSystem.get(Configuration())
    file = fs.globStatus(Path("%s/part*" % hdfs_path))[0].getPath().getName()
    full_path = "%s/%s" % (hdfs_path, filename)
    result = fs.rename(Path("%s/%s" % (hdfs_path, file)), Path(full_path))
    return result


if __name__ == "__main__":
    print("Running Victorian Author analysis...")
    run()
    print("--\n--\nDone!\n--\n--")
