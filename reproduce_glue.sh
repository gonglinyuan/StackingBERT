#!/usr/bin/env bash

BERT_PATH=models/L12/checkpoint_5_280000.pt
DATA_PATH=glue-3
SEED=700

function run_exp {
    TASK_NAME=$1
    TASK_TYPE=$2
    SYMMETRIC_FLAG=$3
    TASK_CRITERION=$4
    N_CLASSES=$5
    N_SENT=$6
    WEIGHT_DECAY=$7
    N_EPOCH=$8
    BATCH_SZ=$9
    LR=${10}
    SEED2=${11}

    # Runs on 1 GPU
    SENT_PER_GPU=$(( BATCH_SZ / 1 ))
    N_UPDATES=$(( ((N_SENT + BATCH_SZ - 1) / BATCH_SZ) * N_EPOCH ))
    WARMUP_UPDATES=$(( (N_UPDATES + 5) / 10 ))

    mkdir -p models/fairseq-${TASK_NAME}
    python train.py data-bin/${DATA_PATH}/${TASK_NAME} --task ${TASK_TYPE} ${SYMMETRIC_FLAG} \
    --arch transformer_classifier_base --n-classes ${N_CLASSES} --load-bert checkpoint.pt \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay ${WEIGHT_DECAY} \
    --lr ${LR} --lr-scheduler linear --warmup-init-lr 1e-07 --warmup-updates ${WARMUP_UPDATES} --min-lr 1e-09 \
    --criterion ${TASK_CRITERION} \
    --max-sentences ${SENT_PER_GPU} --max-update ${N_UPDATES} --seed ${SEED2} \
    --save-dir models/fairseq-${TASK_NAME} --no-progress-bar --no-epoch-checkpoints

    python inference.py data-bin/${DATA_PATH}/${TASK_NAME} --gen-subset test --task ${TASK_TYPE} \
    --path models/fairseq-${TASK_NAME}/checkpoint_last.pt --output predictions/prediction_${TASK_NAME}.txt
}

cd examples/glue

bash process_glue.sh ../../data-bin/bert_corpus/bpe-code ../../data-bin/bert_corpus/dict.txt 3

mkdir -p predictions

echo 'To reproduce our result, please run in 1 GPU'

run_exp 'CoLA' 'glue_single' '' 'cross_entropy_classify_binary' 1 8551 0.01 3 16 0.00005 ${SEED}

run_exp 'MRPC' 'glue_pair' '--symmetric' 'cross_entropy_classify_binary' 1 3668 0.01 3 16 0.00005 ${SEED}

run_exp 'STS-B' 'glue_pair' '--symmetric' 'mean_squared_error' 1 5749 0.01 4 32 0.00005 ${SEED}

run_exp 'RTE' 'glue_pair' '' 'cross_entropy_classify' 2 2475 0.01 4 16 0.00003 ${SEED}

run_exp 'SST-2' 'glue_single' '' 'cross_entropy_classify' 2 67349 0.01 3 16 0.00005 ${SEED}

run_exp 'MNLI' 'glue_pair' '' 'cross_entropy_classify' 3 392702 0.01 3 16 0.00005 ${SEED}

run_exp 'MNLI-mm' 'glue_pair' '' 'cross_entropy_classify' 3 392702 0.01 3 16 0.00005 ${SEED}

run_exp 'QQP' 'glue_pair' '--symmetric' 'cross_entropy_classify_binary' 1 363849 0.01 4 32 0.00005 ${SEED}

run_exp 'QNLI' 'glue_pair' '' 'cross_entropy_classify' 2 108436 0.01 3 32 0.00005 ${SEED}

python inference.py data-bin/${DATA_PATH}/diagnostic --gen-subset test --task glue_pair \
--path models/fairseq-MNLI/checkpoint_last.pt --output predictions/prediction_diagnostic.txt

python examples/process_predictions.py predictions --output predictions
zip predictions.zip predictions/*.tsv
