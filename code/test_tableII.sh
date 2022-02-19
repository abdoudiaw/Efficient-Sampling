#!/bin/bash
#
# Specify the benchmark function and sampler for the test
export BenchmarkFunction='ackley'   # options: ackley, branins, hartmann6, michal5, rosen3, rosen8
export Sampler_DIR='fmin'          # options: fmin, powell
#
#
##### DO NOT MODIFY THESE LINES #####
export Table_DIR=tableII
export Results_DIR=../results #/tableII

cd ${Results_DIR}
mkdir "${BenchmarkFunction}"
cd    ${BenchmarkFunction}
mkdir "${Sampler_DIR}"
cd    "${Sampler_DIR}"
# #
cp ../../../code/${Table_DIR}/${BenchmarkFunction}/${Sampler_DIR}/* .

# run test
echo  "Running ${BenchmarkFunction} with sampler ${Sampler_DIR}"
python main_workflow.py
#
echo "Finished test table I. Cleaning up"

