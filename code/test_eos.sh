#!/bin/bash
#
export testcase='eos'
export Sampler_DIR='powell'          # options: fmin, powell
#
#
##### DO NOT MODIFY THESE LINES #####
export Results_DIR=../results #/tableII

cd ${Results_DIR}
mkdir "${testcase}"
cd    ${testcase}
mkdir "${Sampler_DIR}"
cd    "${Sampler_DIR}"
# #
#cp ../../../code/${Table_DIR}/${testcase}/${Sampler_DIR}/* .
cp /Users/diaw/Efficient-Sampling/code/${testcase}/${Sampler_DIR}/* .

# run test
echo  "Running ${testcase} with sampler ${Sampler_DIR}"
python main_workflow.py
#
echo "Finished running!"


