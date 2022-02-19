#!/bin/bash
#
# Specify the eos model
export EosModel='Skyrme'           # options: Skyrme
#
##### DO NOT MODIFY THESE LINES #####
export EOS_DIR='eos'
export Results_DIR=../results #

cd ${Results_DIR}
mkdir "${EosModel}"
cd    ${EosModel}

# #
cp -r ../../../code/${EOS_DIR}/${EosModel}/* .
# run test
echo  "Running ${EosModel}"
python main_workflow.py
#
echo "Finished learning eos. Cleaning up"


