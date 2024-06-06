export HADOOP_VERSION=3.3.6
export HADOOP_HOME=/local/Hadoop/hadoop-$HADOOP_VERSION
export SPARK_HOME=/local/spark
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
export JAVA_HOME="/usr/pkg/java/sun-8"
export PATH=${PATH}:$JAVA_HOME:$HADOOP_HOME/bin:$SPARK_HOME/bin
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server

