#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6
export PYTHONIOENCODING=utf-8

HOME_DIR='/home/diwu/kpgen/KPEval/'
export PYTHONPATH=${PYTHONPATH}:${HOME_DIR}
mkdir -p ./outputs


# model="17_scibart-large+oagkx"  # 3_multipartiterank, 8_catseqtg+2rf1, 13_settrans, 16_keybart, 17_scibart-large+oagkx


dataset="kp20k"  # kp20k, kptimes

if [[ $dataset =~ ^(kp20k|inspec|krapivin|nus|semeval)$ ]] ; then
    inp_file="/home/diwu/lm-kpgen/DeepKPG/data/scikp/${dataset}/fairseq/test.source"
    ref_file="/home/diwu/lm-kpgen/DeepKPG/data/scikp/${dataset}/fairseq/test.target"
else
    inp_file="/home/diwu/lm-kpgen/DeepKPG/data/${dataset}/fairseq/test.source"
    ref_file="/home/diwu/lm-kpgen/DeepKPG/data/${dataset}/fairseq/test.target"
fi

# pred_file="/Users/xueerli/Desktop/capstone/results/task_definition/kp20k_zeroshot_30_cases_prompt_5.txt"


metrics="retrieval"   # rouge,exact_matching,bert_score,meteor,semantic_matching,retrieval
num_case=30

# 5 prompts
for i in 0 1 2 3 4 5
do
    pred_file="./task_definition/kp20k_zeroshot_30_cases_prompt_${i}.txt"
    model="kp20k_zeroshot_prompt_${i}"
    python3 ${HOME_DIR}/source/run_evaluation.py \
    --config-file config_kp20k_task_def.gin \
    --metrics ${metrics} \
    --input-file ${inp_file} \
    --label-file ${ref_file} \
    --output-file ${pred_file} \
    --log-file-prefix  outputs/${dataset}_${model} \
    --only-use-first-n ${num_case}
done
