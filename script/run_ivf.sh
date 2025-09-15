#!/bin/bash

# DEBUG=1
# Define arrays for dataset, B values, and group
datasets=("gist")
B_values=(32)
group=4096

args_list+=("-enable_PCA=false")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Loop through each B value
    for B in "${B_values[@]}"; do
        for args in "${args_list[@]}"; do
            echo "==========>  Running ./bin/test_ivf -dataset "$dataset" -K "$group" -B "$B" $args ..."
            ./bin/test_ivf -dataset "$dataset" -K "$group" -B "$B" $args
            if [ $? -ne 0 ]; then
                echo "Error: test_ivf failed with exit code $?. Exiting."
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
