need java

MY_USERNAME=$(whoami)
hadoop fs -mkdir -p /user/$MY_USERNAME
hadoop fs -put kdd.data.txt /user/$MY_USERNAME/kdd.data.txt
# Remove output dir if exists
hadoop fs -test -d /user/$MY_USERNAME/vic-output && hadoop fs -rm -r /user/$MY_USERNAME/vic-output

spark-submit --master yarn --deploy-mode cluster --py-files main.py,pyspark_funcs.py main.py $MY_USERNAME

# Print the output (its saved in hdfs)
hadoop fs -cat /user/$MY_USERNAME/vic-output/part-00000
