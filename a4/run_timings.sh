#!/bin/bash

# Variables
EXECUTABLE=jacobi
INPUT_SMALL=test_small.txt
INPUT_LARGE=test_large.txt
OUTPUT_FILE=timing_results.txt
ITERATIONS=("1" "5" "10" "50" "100")

# Compile the CUDA program
echo "Compiling CUDA program..."
make

# Header for the output file
echo "Jacobi Solver Timing Results" > $OUTPUT_FILE
echo "====================================" >> $OUTPUT_FILE

# Function to run tests and collect timings
run_test() {
    local input_file=$1
    local size_label=$2

    echo "Running tests for $size_label grid..." 
    echo -e "\n=== $size_label Grid ===" >> $OUTPUT_FILE

    for iter in "${ITERATIONS[@]}"; do
        echo "Testing $iter iterations..."
        # Run the solver and extract the timing data
        RESULT=$(./$EXECUTABLE -T -i $iter $input_file | grep "Total computation time")
        AVG_TIME=$(./$EXECUTABLE -T -i $iter $input_file | grep "Average time per iteration")
        
        # Log the results to the file
        echo "Iterations: $iter" >> $OUTPUT_FILE
        echo "$RESULT" >> $OUTPUT_FILE
        echo "$AVG_TIME" >> $OUTPUT_FILE
        echo "------------------------------" >> $OUTPUT_FILE
    done
}

# Run tests for both small and large grids
run_test $INPUT_SMALL "Small (10x10)"
run_test $INPUT_LARGE "Large (1024x1024)"

echo "Timing results saved to $OUTPUT_FILE"
