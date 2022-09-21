#!/bin/bash
g++ -o gen_data apex-map.cpp -std=c++11
for i in {1..10000}
do 
    reuse_rate=$((1 + $RANDOM % 98))
    vec_len=$((1 + $RANDOM % 32))
    gen_len=128
    seed=$i
    echo $seed
    ./gen_data $reuse_rate $vec_len $gen_len $seed >> train_data/trace_128_${reuse_rate}_${vec_len}_${gen_len}_${seed}.log
done

