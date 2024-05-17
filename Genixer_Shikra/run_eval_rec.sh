accelerate launch --num_processes 8 \
    --main_process_port 23782 \
    mllm/pipeline/finetune.py \
    config/eval_multi_rec.py \
    --cfg-options model_args.model_name_or_path=checkpoints/shikra-Genixer-350K-7b \
    training_args.output_dir=results/shikra-Genixer-350K-7b