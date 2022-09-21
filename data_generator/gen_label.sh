#!/bin/bash
g++ -o gen_label gen_locality_label.cpp
for FILE in train_data/*
do 
    echo $FILE
    arrIN=(${FILE//// })
    echo ${arrIN[2]}
    ./gen_label $FILE
done
