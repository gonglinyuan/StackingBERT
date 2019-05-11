#!/usr/bin/env bash

cd examples/common

echo ' - Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo ' - Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

echo ' - Cloning FastBPE repository (for faster BPE pre-processing)...'
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 -march=native fastBPE/main.cc -IfastBPE -o fast
cd ..

cd ../..
