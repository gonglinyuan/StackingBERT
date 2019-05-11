# Introduction

This repository is the code to reproduce the result of *Efficient Training of BERT by Progressively Stacking*.
The code is based on [Fairseq](https://github.com/pytorch/fairseq).

# Requirements and Installation
* [PyTorch](http://pytorch.org/) >= 1.0.0
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.7

After PyTorch is installed, you can install requirements with:
```bash
pip install -r requirements.txt
```

# Getting Started

## Step 1:

```bash
bash install.sh
```

This script downloads:

1. [Moses Decoder](https://github.com/moses-smt/mosesdecoder)
1. [Subword NMT](https://github.com/rsennrich/subword-nmt)
1. [Fast BPE](https://github.com/glample/fastBPE) (In the next steps, we use Subword NMT instead of Fast BPE. Recommended if you want to generate your own dictionary on a large-scale dataset.)

These library will do cleaning, tokenization, and BPE encoding for GLUE data in step 3. They will also be helpful if you want to make your own corpus for BERT training or if you want to test our model on your own tasks.
## Step 2: 

```bash
bash reproduce_bert.sh
```

This script runs progressive stacking and train a BERT. The code is tested on 4 Tesla P40 GPUs (24GB Gmem). For different hardware, you probably need to change the maximum number of tokens per batch (by changing `max-tokens` and `update-freq`).

## Step 3:

```bash
bash reproduce_glue.sh
```

This script fine-tunes the BERT trained in step 2. The script chooses the checkpoint trained for 400K steps, which is the same as the *stacking* model in our paper.
