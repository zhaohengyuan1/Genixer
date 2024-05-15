accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/pretrain_final55_stage2.py \
        --cfg-options model_args.model_name_or_path=exp/genixer_shikra_pretrain_stage1 \
        training_args.output_dir=exp/genixer_shikra_pretrain_stage2