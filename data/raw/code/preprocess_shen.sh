#!/bin/bash

set -e
set -u

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do
  for run in 1 2 3 4 5 6 7 8; do
    code/vol2shenroi_ts $sub $run
  done
done
