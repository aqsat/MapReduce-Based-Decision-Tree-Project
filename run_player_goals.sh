#!/bin/bash

# Paths to scripts and files
MAPPER_PATH="/home/seed/PROJECT/mapper_reducer/player_mapper.py"
REDUCER_PATH="/home/seed/PROJECT/mapper_reducer/player_reducer.py"
HDFS_INPUT_DIR="/user/hadoop/input/WorldCupPlayers.csv"
HDFS_OUTPUT_DIR="/user/hadoop/output/player_goals_output"
LOCAL_OUTPUT_DIR="/home/seed/PROJECT/output"
LOCAL_OUTPUT_FILE="$LOCAL_OUTPUT_DIR/player_goals_output.txt"
HADOOP_STREAMING_JAR="/home/seed/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.4.jar"

# Step 1: Verify script permissions
chmod +x $MAPPER_PATH $REDUCER_PATH

# Step 2: Create local output directory
echo "Creating local output directory..."
mkdir -p $LOCAL_OUTPUT_DIR

# Step 3: Remove existing output directory in HDFS
echo "Removing existing HDFS output directory..."
hdfs dfs -rm -r -f $HDFS_OUTPUT_DIR 2>/dev/null

# Step 4: Verify input file exists in HDFS
echo "Checking input file in HDFS..."
if ! hdfs dfs -test -e $HDFS_INPUT_DIR; then
    echo "Error: Input file $HDFS_INPUT_DIR does not exist in HDFS."
    exit 1
fi

# Step 5: Run the MapReduce job
echo "Running Hadoop MapReduce job..."
hadoop jar $HADOOP_STREAMING_JAR \
    -files $MAPPER_PATH,$REDUCER_PATH \
    -mapper "python3 player_mapper.py" \
    -reducer "python3 player_reducer.py" \
    -input $HDFS_INPUT_DIR \
    -output $HDFS_OUTPUT_DIR

# Step 6: Check job status and retrieve output
if [ $? -eq 0 ]; then
    echo "MapReduce job completed successfully."
    echo "Output directory contents in HDFS:"
    hdfs dfs -ls $HDFS_OUTPUT_DIR

    # Copy output to local directory
    echo "Copying output to local directory: $LOCAL_OUTPUT_FILE"
    hdfs dfs -getmerge $HDFS_OUTPUT_DIR/part-* $LOCAL_OUTPUT_FILE
    if [ $? -eq 0 ]; then
        echo "Output successfully saved to $LOCAL_OUTPUT_FILE"
        head -n 10 $LOCAL_OUTPUT_FILE
    else
        echo "Error: Failed to copy output to local directory."
        exit 1
    fi

    # Check mapper/reducer logs
    echo "Checking mapper/reducer logs..."
    LATEST_APP_ID=$(yarn application -list -appStates FINISHED | grep "hadoop" | head -n 1 | awk '{print $1}')
    if [ -n "$LATEST_APP_ID" ]; then
        yarn logs -applicationId $LATEST_APP_ID | grep DEBUG > $LOCAL_OUTPUT_DIR/debug_logs.txt
        echo "Debug logs saved to $LOCAL_OUTPUT_DIR/debug_logs.txt"
        head -n 20 $LOCAL_OUTPUT_DIR/debug_logs.txt
    else
        echo "Warning: Could not find application ID for logs."
    fi
else
    echo "Error: MapReduce job failed. Check Hadoop logs for details."
    exit 1
fi