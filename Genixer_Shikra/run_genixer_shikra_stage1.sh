accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/pretrain_concat8_stage1.py \
        --cfg-options model_args.model_name_or_path=exp/genixer_shikra_pretrain_stage0 \
        training_args.output_dir=exp/genixer_shikra_pretrain_stage1