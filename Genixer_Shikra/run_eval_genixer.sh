accelerate launch --num_processes 8 \
    --main_process_port 23782 \
    mllm/pipeline/finetune.py \
    config/genixer_eval_GenQA.py \
    --cfg-options model_args.model_name_or_path=checkpoints/Genixer-shikra-7b \
    training_args.output_dir=results/Genixer-shikra-7b
