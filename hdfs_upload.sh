#!/bin/bash

# Define the path to the local data directory
LOCAL_DATA_PATH="/home/seed/PROJECT/Data"

# Define the HDFS input directory
HDFS_INPUT_DIR="/user/hadoop/input"

# Create the HDFS input directory if it doesn't exist
hdfs dfs -mkdir -p $HDFS_INPUT_DIR

# Upload the CSV files to HDFS
hdfs dfs -put $LOCAL_DATA_PATH/WorldCupMatches.csv $HDFS_INPUT_DIR/
hdfs dfs -put $LOCAL_DATA_PATH/WorldCupPlayers.csv $HDFS_INPUT_DIR/
hdfs dfs -put $LOCAL_DATA_PATH/WorldCups.csv $HDFS_INPUT_DIR/

# List the uploaded files to confirm
hdfs dfs -ls $HDFS_INPUT_DIR

