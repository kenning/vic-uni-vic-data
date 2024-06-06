import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, NGram
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier


username = sys.argv[1]
test_run = False
if username == "test":
    test_run = True
    print("Test run")
else:
    print(f"Running as user {username}")

small_sample_size = False
sample_fraction = 0.0001
if len(sys.argv) > 2:
    if sys.argv[2] == "--sample":
        small_sample_size = True

        print(f"Only running on sample of {sample_fraction*100:02}% of data")


def run():
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
    if small_sample_size:
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
    trigram = NGram(n=3, inputCol="filtered", outputCol="trigrams")

    unigram_vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
    bigram_vectorizer = CountVectorizer(inputCol="bigrams", outputCol="features")
    trigram_vectorizer = CountVectorizer(inputCol="trigrams", outputCol="features")

    ############################################################
    # Run DT and LR models
    ############################################################

    kwargz = {
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

    # TODO TEMP
    # lr_logs, _, _ = build_pipeline_and_evaluate(**kwargz)
    lr_logs = ["lr log 1", "lr log 2"]

    kwargz["is_dt"] = True
    dt_logs, dt_strings, feature_dfs = build_pipeline_and_evaluate(**kwargz)

    ############################################################
    # Print results, optionally to HDFS
    ############################################################

    all_logs = lr_logs + dt_logs + dt_strings
    print("\n".join(all_logs))

    if not test_run:
        rdd = sc.parallelize(lr_logs + dt_logs, numSlices=1)
        output_path = f"hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output"
        rdd.saveAsTextFile(output_path)

        for i, df in enumerate(feature_dfs):
            df_output_path = f"{output_path}/feature_df_{i}"
            df.coalesce(1).write.mode("overwrite").csv(df_output_path, header=True)

    print("\n\nDone!\n\n")


############################################################
# Builds either LR or DT pipeline and evaluates data.
# Also generates decision tree, as well as ngram dataframe.
############################################################
def build_pipeline_and_evaluate(
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
    lr = LogisticRegression(featuresCol="features", labelCol="author")
    dt = DecisionTreeClassifier(labelCol="author", featuresCol="features")
    classifier = dt if is_dt else lr
    classifier_name = "Decision Tree" if is_dt else "Logistic Regression"
    result_strings = []
    decision_tree_strings = []
    feature_dfs = []

    for i in range(3):
        curr_pipeline_name = ""
        if i == 0:
            curr_pipeline = Pipeline(
                stages=[tokenizer, remover, unigram_vectorizer, classifier]
            )
            curr_pipeline_name = "Unigram"
        elif i == 1:
            curr_pipeline = Pipeline(
                stages=[tokenizer, remover, bigram, bigram_vectorizer, classifier]
            )
            curr_pipeline_name = "Bigram"
        elif i == 2:
            curr_pipeline = Pipeline(
                stages=[tokenizer, remover, trigram, trigram_vectorizer, classifier]
            )
            curr_pipeline_name = "Trigram"
        else:
            raise Exception("this should not happen")

        curr_model = curr_pipeline.fit(training_data)
        curr_predictions = curr_model.transform(test_data)

        curr_evaluator = MulticlassClassificationEvaluator(
            labelCol="author", predictionCol="prediction", metricName="accuracy"
        )
        curr_accuracy = curr_evaluator.evaluate(curr_predictions)
        result_string = f"[{classifier_name}] Test Accuracy for {curr_pipeline_name}: {curr_accuracy:.5f}"
        print(result_string)
        result_strings.append(result_string)

        if is_dt:
            dt_str, feature_df = create_dt_and_feature_df(dt_model=curr_model)
            decision_tree_strings.append(dt_str)
            feature_dfs.append(feature_df)

    return result_strings, decision_tree_strings, feature_dfs


############################################################
# Generate decision tree and ngram dataframe.
############################################################
def create_dt_and_feature_df(dt_model):
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

        # Use regular expressions to find and replace feature indices
        tree_string_with_names = re.sub(r"feature (\d+)", replace_match, tree_string)
        return tree_string_with_names, feature_indices

    tree_string_with_names, feature_indices = replace_and_collect_feature_indices(
        tree_string, dt_vocabulary
    )

    # Count occurrences of each feature index
    feature_counts = Counter(feature_indices)

    # Create a list of tuples (feature_name, count)
    feature_usage_data = [
        (dt_vocabulary[idx], count) for idx, count in feature_counts.items()
    ]

    # Initialize SparkSession (if not already initialized)
    spark = SparkSession.builder.appName("FeatureUsage").getOrCreate()

    # Create a DataFrame from the feature usage data
    feature_usage_df = spark.createDataFrame(
        feature_usage_data, ["feature_name", "count"]
    )

    # Show the DataFrame
    feature_usage_df.orderBy(col("count").desc()).show()

    return tree_string_with_names, feature_usage_df


if __name__ == "__main__":
    print("Running Victorian Author analysis...")
    run()
