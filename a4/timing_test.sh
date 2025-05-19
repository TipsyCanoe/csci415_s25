#!/bin/bash

echo "Grid Size	Iterations	Total Time (ms)	Time Per Iteration (ms)"

for i in 1 5 10 50 100; do
  RESULTS=$(./jacobi -T -i $i test_small.txt 2>&1)
  TOTAL_TIME=$(echo "$RESULTS" | grep "Total computation time" | cut -d':' -f2 | tr -d ' ')
  AVG_TIME=$(echo "$RESULTS" | grep "Average time per iteration" | cut -d':' -f2 | tr -d ' ')
  echo "10 x 10	$i	$TOTAL_TIME	$AVG_TIME"
done

for i in 1 5 10 50 100; do
  RESULTS=$(./jacobi -T -i $i test_large.txt 2>&1)
  TOTAL_TIME=$(echo "$RESULTS" | grep "Total computation time" | cut -d':' -f2 | tr -d ' ')
  AVG_TIME=$(echo "$RESULTS" | grep "Average time per iteration" | cut -d':' -f2 | tr -d ' ')
  echo "1024 x 1024	$i	$TOTAL_TIME	$AVG_TIME"
done