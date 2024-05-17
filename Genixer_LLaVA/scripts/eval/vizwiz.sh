#!/bin/bash

modelname=llava-Genixer-915K-FT-8K-v1.5-7b

python -m llava.eval.model_vqa_loader \
    --model-path exp/$modelname \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /dataset/lavis/vizwiz/test/ \
    --answers-file ./playground/data/eval/vizwiz/answers/$modelname.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$modelname.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$modelname.json
