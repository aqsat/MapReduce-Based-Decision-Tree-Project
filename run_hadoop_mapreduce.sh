# #!/bin/bash

# # Paths to scripts and files
# MAPPER_PATH="/home/seed/PROJECT/mapper_reducer/mapper.py"
# REDUCER_PATH="/home/seed/PROJECT/mapper_reducer/reducer.py"
# HDFS_INPUT_DIR="/user/hadoop/input/WorldCupMatches.csv"
# HDFS_OUTPUT_DIR="/user/hadoop/output"
# HADOOP_STREAMING_JAR="/home/seed/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.4.jar"

# # Step 1: Verify script permissions
# chmod +x $MAPPER_PATH $REDUCER_PATH

# # Step 2: Remove existing output directory in HDFS
# echo "Removing existing HDFS output directory..."
# hdfs dfs -rm -r -f $HDFS_OUTPUT_DIR 2>/dev/null

# # Step 3: Verify input file exists in HDFS
# echo "Checking input file in HDFS..."
# if ! hdfs dfs -test -e $HDFS_INPUT_DIR; then
#     echo "Error: Input file $HDFS_INPUT_DIR does not exist in HDFS."
#     exit 1
# fi

# # Step 4: Run the MapReduce job
# echo "Running Hadoop MapReduce job..."
# hadoop jar $HADOOP_STREAMING_JAR \
#     -file $MAPPER_PATH -mapper "python3 mapper.py" \
#     -file $REDUCER_PATH -reducer "python3 reducer.py" \
#     -input $HDFS_INPUT_DIR \
#     -output $HDFS_OUTPUT_DIR

# # Step 5: Check job status and display output
# if [ $? -eq 0 ]; then
#     echo "MapReduce job completed successfully."
#     echo "Output directory contents:"
#     hdfs dfs -ls $HDFS_OUTPUT_DIR
#     echo "Sample output:"
#     hdfs dfs -cat $HDFS_OUTPUT_DIR/part-00000 | head
# else
#     echo "Error: MapReduce job failed. Check Hadoop logs for details."
#     exit 1
# fi

#!/bin/bash

# Paths to scripts and files
BASE_PATH="/home/seed/PROJECT/mapper_reducer"
HDFS_BASE_INPUT="/user/hadoop/input/WorldCupMatches_clean.csv"  
HDFS_BASE_OUTPUT="/user/hadoop/output"
LOCAL_OUTPUT_DIR="/home/seed/PROJECT/output"
HADOOP_STREAMING_JAR="/home/seed/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.4.jar"
# Job configurations
JOBS=(
    "matches_played:matches_played_mapper.py:matches_played_reducer.py:team_matches.txt"
    "outcomes:outcomes_mapper.py:outcomes_reducer.py:team_outcomes.txt"
    "avg_goals:avg_goals_mapper.py:avg_goals_reducer.py:team_avg_goals.txt"
)

# Step 1: Create local output directory
echo "Creating local output directory..."
mkdir -p $LOCAL_OUTPUT_DIR

# Step 2: Enable log aggregation
export HADOOP_LOG_DIR=/tmp/hadoop-logs
hdfs dfs -mkdir -p /user/hadoop/logs

# Step 3: Verify input file exists in HDFS
echo "Checking input file in HDFS..."
if ! hdfs dfs -test -e $HDFS_BASE_INPUT; then
    echo "Error: Input file $HDFS_BASE_INPUT does not exist in HDFS."
    exit 1
fi

# Step 4: Run each MapReduce job
for job in "${JOBS[@]}"; do
    IFS=':' read -r job_name mapper reducer output_file <<< "$job"
    MAPPER_PATH="$BASE_PATH/$mapper"
    REDUCER_PATH="$BASE_PATH/$reducer"
    HDFS_OUTPUT_DIR="$HDFS_BASE_OUTPUT/$job_name"
    LOCAL_OUTPUT_FILE="$LOCAL_OUTPUT_DIR/$output_file"

    # Verify script permissions
    chmod +x $MAPPER_PATH $REDUCER_PATH

    # Remove existing output directory in HDFS
    echo "Removing existing HDFS output directory for $job_name..."
    hdfs dfs -rm -r -f $HDFS_OUTPUT_DIR 2>/dev/null

    # Run the MapReduce job
    echo "Running Hadoop MapReduce job for $job_name..."
    hadoop jar $HADOOP_STREAMING_JAR \
        -D yarn.log-aggregation-enable=true \
        -files $MAPPER_PATH#$mapper,$REDUCER_PATH#$reducer \
        -mapper "python3 $mapper" \
        -reducer "python3 $reducer" \
        -input $HDFS_BASE_INPUT \
        -output $HDFS_OUTPUT_DIR \
        -cmdenv PYTHONPATH=/usr/bin/python3

    # Check job status and retrieve output
    if [ $? -eq 0 ]; then
        echo "MapReduce job for $job_name completed successfully."
        echo "Output directory contents in HDFS:"
        hdfs dfs -ls $HDFS_OUTPUT_DIR

        # Copy output to local directory
        echo "Copying output to local directory: $LOCAL_OUTPUT_FILE"
        hdfs dfs -getmerge $HDFS_OUTPUT_DIR/part-00000 $LOCAL_OUTPUT_FILE
        if [ $? -eq 0 ]; then
            echo "Output successfully saved to $LOCAL_OUTPUT_FILE"
            head -n 10 $LOCAL_OUTPUT_FILE
        else
            echo "Error: Failed to copy output to local directory for $job_name."
            exit 1
        fi
    else
        echo "Error: MapReduce job for $job_name failed. Check Hadoop logs for details."
        exit 1
    fi
done

# Step 5: Check mapper/reducer logs
echo "Checking mapper/reducer logs..."
LATEST_APP_IDS=$(yarn application -list -appStates FINISHED | grep "application_1746060620948" | awk '{print $1}')
if [ -n "$LATEST_APP_IDS" ]; then
    for APP_ID in $LATEST_APP_IDS; do
        echo "Retrieving logs for $APP_ID..."
        yarn logs -applicationId $APP_ID | grep DEBUG >> $LOCAL_OUTPUT_DIR/debug_logs.txt 2>/dev/null
    done
    echo "Debug logs saved to $LOCAL_OUTPUT_DIR/debug_logs.txt"
    if [ -s $LOCAL_OUTPUT_DIR/debug_logs.txt ]; then
        head -n 20 $LOCAL_OUTPUT_DIR/debug_logs.txt
    else
        echo "No DEBUG messages found in logs."
    fi
else
    echo "Warning: Could not find application IDs for logs."
fi