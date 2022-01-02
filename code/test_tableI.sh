#!/bin/bash
#
# Specify the benchmark function and sampler for the test
export Function='easom'           # options: easom, hartmann6, michal2, rast, rosen, rosen8
export Sampler_DIR='looseopt'          # options: looseopt, loosernd, tightopt,tightrnd
#
#
##### DO NOT MODIFY THESE LINES #####
export Table_DIR=${Efficient-Sampling/tableI}
export Results_DIR=${Efficient-Sampling/results/tableI}
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
echo "Finished test table II. Cleaning up"
#
cd ${Table_DIR}/..
