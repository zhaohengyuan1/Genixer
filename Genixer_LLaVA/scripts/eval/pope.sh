#!/bin/bash

modelname=llava-Genixer-915K-FT-8K-v1.5-7b

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/$modelname \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /yourpath/coco2014/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$modelname.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$modelname.jsonl
