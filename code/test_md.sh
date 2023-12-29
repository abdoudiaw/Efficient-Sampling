#!/bin/bash
#
# Specify the eos model
export PlasmaModel='ocp'           # options: Skyrme
#
# Install LAMMPS
export lammps_DIR="lammps/src" #
cd ${lammps_DIR}

make yes-kspace
make mpi


##### DO NOT MODIFY THESE LINES #####


export MD_DIR='md'
export Results_DIR=../results #

cd ${Results_DIR}
mkdir "${PlasmaModel}"
cd    ${PlasmaModel}

#LAMMPS executable to test directory
cp -r ../../../code/${lammps_DIR}/lmp_mpi .

# #
#cp -r /Users/diaw/Efficient-Sampling/code/${MD_DIR}/${PlasmaModel}/* .
cp -r ../../../code/${MD_DIR}/${PlasmaModel}/* .
# run test
echo  "Running ${PlasmaModel}"
python main_workflow.py
#
echo "Finished learning eos. Cleaning up"


