import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, NGram
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier


username = sys.argv[1]
print(username)

small_sample_size = False
if len(sys.argv) > 2:
    if sys.argv[2] == "--sample":
        small_sample_size = True
print(small_sample_size)


def test(a, b, c):
    print(f"{a=} {b=} {c=}")


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
            "data/dataset/Gungor_2018_VictorianAuthorAttribution_data.csv",
        )
    )

    data = data.withColumn("author", col("author").cast(IntegerType()))

    # For local testing: take a small subset of data
    if small_sample_size:
        sample = data.sample(withReplacement=False, fraction=0.002, seed=100)
    else:
        sample = data

    (training_data, test_data) = sample.randomSplit([0.8, 0.2], seed=100)

    # print(sample.printSchema())
    # sample.show(5)

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

    # num_iterations = 10
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

    lr_logs = build_pipeline_and_evaluate(**kwargz)

    kwargz["is_dt"] = True
    dt_logs = build_pipeline_and_evaluate(**kwargz)

    print("\n".join(lr_logs + dt_logs))

    rdd = sc.parallelize(lr_logs + dt_logs, numSlices=1)
    output_path = f"hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output"
    rdd.saveAsTextFile(output_path)


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

    return result_strings


if __name__ == "__main__":
    print("Running Victorian Author analysis...")
    run()
