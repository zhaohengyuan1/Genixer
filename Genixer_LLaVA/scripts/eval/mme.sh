# !/bin/bash

modelname=llava-v1.5-7b

modelname=llava-Genixer-915K-FT-8K-v1.5-7b

python -m llava.eval.model_vqa_loader \
    --model-path exp/$modelname \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$modelname.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $modelname

cd eval_tool

python calculation.py --results_dir answers/$modelname
