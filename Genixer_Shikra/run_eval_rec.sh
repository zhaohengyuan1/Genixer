accelerate launch --num_processes 8 \
    --main_process_port 23782 \
    mllm/pipeline/finetune.py \
    config/eval_multi_rec.py \
    --cfg-options model_args.model_name_or_path=exp/genixer_shikra_pretrain_stage2 \
    training_args.output_dir=results/genixer_shikra_pretrain_stage2
