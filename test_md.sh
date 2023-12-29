#!/bin/bash
#
# Specify the eos model
export PlasmaModel='ocp'           # options: Skyrme
#
##### DO NOT MODIFY THESE LINES #####
export MD_DIR='md'
export Results_DIR=../results #

cd ${Results_DIR}
mkdir "${PlasmaModel}"
cd    ${PlasmaModel}

# #
#cp -r /Users/diaw/Efficient-Sampling/code/${MD_DIR}/${PlasmaModel}/* .
cp -r ../../../code/${MD_DIR}/${PlasmaModel}/* .
# run test
echo  "Running ${PlasmaModel}"
python main_workflow.py
#
echo "Finished learning eos. Cleaning up"


