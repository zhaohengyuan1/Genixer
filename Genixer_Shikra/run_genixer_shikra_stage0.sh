accelerate launch --num_processes 8 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/pretrain_concat3_stage0.py \
        --cfg-options model_args.model_name_or_path=checkponts/vicuna-7b-v1.1 \
        training_args.output_dir=exp/genixer_shikra_pretrain_stage0