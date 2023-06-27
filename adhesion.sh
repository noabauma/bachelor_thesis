#!/bin/bash 
for i in {80..120..1} 
do 
	echo "epsilon: " $i
    mpirun -np 2 python3 adhesion.py $i 
done 