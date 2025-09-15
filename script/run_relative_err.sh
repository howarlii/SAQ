#!/bin/bash

# DEBUG=1
# Define arrays for dataset, B values, and group
datasets=("gist" "openai1536")
B_values=(4 8 1 2 3 5 6 7)
group=4096

datasets=("gist")
# datasets=("openai1536")
# datasets=("laion")

B_values=(1 4 8)

args_list+=("-enable_segmentation=true")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Loop through each B value
    for B in "${B_values[@]}"; do
        for args in "${args_list[@]}"; do
            echo "==========>  Running ./bin/create_index -dataset "$dataset" -K "$group" -B "$B" $args ..."
            ./bin/create_index -dataset "$dataset" -K "$group" -B "$B" $args
            if [ $? -ne 0 ]; then
                echo "Error: create_index failed with exit code $?. Exiting."
                exit 1
            fi

            echo "==========>  Running ./bin/test_relative_error -dataset "$dataset" -K "$group" -B "$B" $args ..."
            ./bin/test_relative_error -dataset "$dataset" -K "$group" -B "$B" $args
            if [ $? -ne 0 ]; then
                echo "Error: test_relative_error failed with exit code $?. Exiting."
                exit 1
            fi
        done
        if [ "$DEBUG" == 1 ]; then
            echo "Debug flag is set. Breaking the loop."
            exit 0
        fi
    done
done

echo "All commands executed successfully."
