#!/bin/bash
#
# Specify the benchmark function and sampler for the test
export Function='ackley'   # options: ackley, branins, hartmann6, michal5, rosen3, rosen8
export Sampler_DIR='fmin'          # options: fmin, powell
#
#
##### DO NOT MODIFY THESE LINES #####
export Table_DIR=${Efficient-Sampling/tableII}
export Results_DIR=${Efficient-Sampling/results/tableII}
#
cd ${Results_DIR}
mkdir "${Function}"
cd    "${Function}"
mkdir "${Sampler_DIR}"
cd    "${Sampler_DIR}"
#
cp ${Table_DIR}/${Function}/${Sampler_DIR}/* .
#
# run test

echo  "Running ${Function} with sampler ${Sampler_DIR}"

python ${Results_DIR}/${Function}/${Sampler_DIR}/main_workflow.py
#
echo "Finished test table I. Cleaning up"
#
cd ${Table_DIR}/..
