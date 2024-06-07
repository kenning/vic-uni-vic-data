# AIML427 Personal Project: Victorian Author Analysis

## Description

Project by Nick McCormick for analyzing Victorian-era authors. 

Github repo is public: https://github.com/kenning/vic-uni-vic-data

## Getting relevant datasets

Before running submitted code, you should make sure you have the original dataset downloaded to the root directory. This dataset, as well as the preprocessed PCA datasets, can be found here: https://drive.google.com/drive/folders/15KQLnmcjPgDoswjJt3avVMdY9WVM-aLi 

For all programs, you must have “Gungor_2018_VictorianAuthorAttribution_data.csv” downloaded to your root folder. For PCA analysis, you should download all other datasets in the google drive folder and then run pca_main.py (using run_spark.sh).

## Installation steps
- Install PySpark and Numpy by executing `pip install pyspark numpy`
- SSH into the CO246 Hadoop cluster by `ssh co246a-1`
- Sign into the cluster by executing `kinit`
- Source the required Hadoop environmental variables by executing `source hadoop_env.csh`
- Make `run_spark.sh` executable by executing `chmod +x run_spark.sh`
- Submit the program in the cluster by executing `sh run_spark.sh`
- To copy/move the log file from the HDFS to the OS, copy the file `part-0000.txt` from the directory: `hdfs://co246a-a.ecs.vuw.ac.nz:9000/user/{username}/vic-output` using the `-copyToLocal` and `-rm` commands.
