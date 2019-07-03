#!/usr/bin/env bash

mkdir -p data-bin

cd examples/bert

wget -t 0 -c -T 20 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -b 30G -q -o - > enwiki.txt

cat enwiki.txt | \
python ../common/remove_non_utf8_chars.py | \
python ../common/precleanup_english.py | \
perl ../common/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl en | \
perl ../common/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | \
python filter_and_cleanup_lines.py > corpus.cleaned.txt

python split.py corpus.cleaned.txt corpus 15000000

cat corpus.valid.txt | \
python segment_sentence.py | \
../common/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 1 -no-escape -l en | \
gawk '{print tolower($0);}' > corpus.valid.tok

for i in 0 1 2 3
do
cat corpus.train.txt.${i} | \
python segment_sentence.py | \
../common/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 1 -no-escape -l en | \
gawk '{print tolower($0);}' > corpus.train.tok.${i}
done

rm corpus.train.tok ||:
for i in 0 1 2 3; do cat corpus.train.tok.${i} >> corpus.train.tok; done

#../common/fastBPE/fast learnbpe 32640 corpus.train.tok > bpe-code
../common/subword-nmt/subword_nmt/learn_bpe.py -s 32640 < corpus.train.tok > bpe-code

cat corpus.train.tok | \
python concat_short_sentences.py | \
python ../common/length_filter_by_char.py 20 1000000 > corpus.train.tok.tmp
#../common/fastBPE/fast applybpe corpus.train.tok.bpe corpus.train.tok.tmp bpe-code
../common/subword-nmt/subword_nmt/apply_bpe.py -c bpe-code < corpus.train.tok.tmp > corpus.train.tok.bpe
rm corpus.train.tok.tmp

cat corpus.valid.tok | \
python concat_short_sentences.py | \
python ../common/length_filter_by_char.py 20 1000000 > corpus.valid.tok.tmp
#../common/fastBPE/fast applybpe corpus.valid.tok.bpe corpus.valid.tok.tmp bpe-code
../common/subword-nmt/subword_nmt/apply_bpe.py -c bpe-code < corpus.valid.tok.tmp > corpus.valid.tok.bpe
rm corpus.valid.tok.tmp

cd ../..
python preprocess.py --only-source --workers 16 --nwordssrc 32768 \
--trainpref examples/bert/corpus.train.tok.bpe \
--validpref examples/bert/corpus.valid.tok.bpe \
--destdir data-bin/bert_corpus

cp examples/bert/bpe-code data-bin/bert_corpus/

echo 'To reproduce our result, please run in 4 GPUs'

mkdir -p models

python train.py data-bin/bert_corpus --task bert \
--arch transformer_bert_L3_A12 --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
--lr 0.0008 --lr-scheduler linear --warmup-init-lr 1e-07 --warmup-updates 10000 --end-lr 0.00055 --min-lr 1e-09 \
--criterion cross_entropy_bert \
--max-tokens 25600 --max-epoch 1 --max-update 50000 \
--save-dir models/L3 --no-progress-bar --log-interval 100 --save-interval-updates 10000 --keep-interval-updates 5

python double.py models/L3/checkpoint_1_50000.pt checkpoint_double.pt

python train.py data-bin/bert_corpus --task bert --load-bert checkpoint_double.pt --reset-optimizer \
--arch transformer_bert_L6_A12 --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
--lr 0.00055 --lr-scheduler linear --warmup-updates 1 --end-lr 0.0003 --min-lr 1e-09 \
--criterion cross_entropy_bert \
--max-tokens 14336 --update-freq 2 --max-update 70000 --seed 2 \
--save-dir models/L6 --no-progress-bar --log-interval 100 --save-interval-updates 10000 --keep-interval-updates 5

rm checkpoint_double.pt
python double.py models/L6/checkpoint_1_70000.pt checkpoint_double.pt

python train.py data-bin/bert_corpus --task bert --load-bert checkpoint_double.pt --reset-optimizer \
--arch transformer_bert_base --share-all-embeddings \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
--lr 0.0003 --lr-scheduler linear --warmup-updates 1 --min-lr 1e-09 \
--criterion cross_entropy_bert \
--max-tokens 10240 --update-freq 3 --max-update 800000 --seed 3 \
--save-dir models/L12 --no-progress-bar --log-interval 100 --save-interval-updates 10000

rm checkpoint_double.pt
