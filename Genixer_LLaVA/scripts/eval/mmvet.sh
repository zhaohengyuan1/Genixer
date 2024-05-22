#!/bin/bash

modelname=llava-Genixer-915K-FT-8K-v1.5-7b

python -m llava.eval.model_vqa \
    --model-path checkpoints/$modelname \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$modelname.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$modelname.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$modelname.json

