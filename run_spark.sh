need java

MY_USERNAME=$(whoami)
hadoop fs -mkdir -p /user/$MY_USERNAME
hadoop fs -put Gungor_2018_VictorianAuthorAttribution_data.csv /user/$MY_USERNAME/Gungor_2018_VictorianAuthorAttribution_data.csv

# Remove output dir if exists
hadoop fs -test -d /user/$MY_USERNAME/vic-output && hadoop fs -rm -r /user/$MY_USERNAME/vic-output

echo "Running spark-submit"
spark-submit --master yarn --deploy-mode cluster main.py $MY_USERNAME 
# spark-submit --master yarn --deploy-mode cluster dt_or_lr_main.py $MY_USERNAME 
# spark-submit --master yarn --deploy-mode cluster pca_main.py $MY_USERNAME 

# Print the output (its saved in hdfs)
hadoop fs -cat /user/$MY_USERNAME/vic-output/logs/part-00000

# Uncomment if you generated a decision tree dataframe
# echo "--"
# hadoop fs -cat /user/$MY_USERNAME/vic-output/df-0/feat_df_0.csv