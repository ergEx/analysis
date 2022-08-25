#!/bin/sh
start_time=$(date +"%T")
echo "Starting analysis: $start_time \n"
echo "Loading data \n"
#python readingdata.py
echo "Running indifference eta analysis \n"
#python analysis_indifferences.py
end_time=$(date +"%T")

#HLM modelleing not integrated yet!

echo "Done: $end_time"