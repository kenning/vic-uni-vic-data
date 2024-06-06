# AIML427 Personal Project: Victorian Author Analysis

## Description

Project by Nick McCormick for analyzing Victorian-era authors. 

Github repo is public: https://github.com/kenning/vic-uni-vic-data

## Usage

General usage is below, but you can run the program differently:
- To run on one node instead of the cluster, you can just run:
	`spark-submit main.py $(whoami)`
- To run only on a small subset of data, supply "--sample" as a second argument to main.py.
	`spark-submit main.py $(whoami) --sample`

## Installation steps

- Install PySpark and Numpy by executing `pip install pyspark numpy`
- SSH into the CO246 Hadoop cluster by `ssh co246a-1`
- Sign into the cluster by executing `kinit`
- Source the required Hadoop environmental variables by executing `source hadoop_env.csh`
- Make `run_spark.sh` executable by executing `chmod +x run_spark.sh`
- Submit the program in the cluster by executing `sh run_spark.sh`
- To copy/move the log file from the HDFS to the OS, copy the file `part-0000.txt` from the directory: `hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output` using the `-copyToLocal` and `-rm` commands.

## Downloading the dataset

**Note: not relevant for vic uni staff, as the whole project is in a large zip file including dataset.**

Pulling from github can cause issues apparently. Try this:
`wget 'https://media.githubusercontent.com/media/kenning/vic-uni-vic-data/main/data/dataset/dataset.zip`

Then unzip the csv file and move to root directory. (It is gitignored)

